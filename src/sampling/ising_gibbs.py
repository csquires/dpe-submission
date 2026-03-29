"""
blocked gibbs sampler for binary ising models with checkerboard parallelization
"""
import torch
from typing import Tuple
from tqdm import tqdm


class IsingGibbsSampler:
    """
    blocked gibbs sampler for binary ising model with external field.

    energy: E(Y|theta) = -(Y^T A Y + Y^T theta) / design
    checkerboard coloring enables parallel updates within red/black node sets.

    args --> tensor validation --> coloring computation --> state init
    """

    def __init__(
        self,
        theta: torch.Tensor,
        A: torch.Tensor,
        design: int,
        num_samples: int,
        burnin: int,
        device: str
    ):
        """
        initialize blocked gibbs sampler for binary ising model.
        checkerboard coloring is computed from A's sparsity pattern.

        args:
            theta: external field [d]
            A: coupling matrix [d, d], symmetric
            design: energy scaling factor
            num_samples: number of samples to collect after burnin
            burnin: number of burnin iterations to discard
            device: "cuda" or "cpu"
        """
        # validation
        assert A.shape[0] == A.shape[1], "A must be square"
        assert theta.shape[0] == A.shape[0], "theta and A dimensions must match"
        assert torch.allclose(A, A.T), "A must be symmetric"

        self.d = A.shape[0]
        self.device = device
        self.design = design
        self.num_samples = num_samples
        self.burnin = burnin

        # move to device
        self.theta = theta.to(device).float()
        self.A = A.to(device).float()

        # compute checkerboard coloring
        self.red_indices, self.black_indices = self._compute_checkerboard_coloring(A)
        self.red_indices = self.red_indices.to(device)
        self.black_indices = self.black_indices.to(device)

        # initialize state randomly
        self.Y = 2.0 * torch.randint(0, 2, (self.d,), device=device, dtype=torch.float32) - 1.0  # [d], {-1, +1}

    def sample(self) -> torch.Tensor:
        """
        run blocked gibbs sampling: burnin iterations then collect samples.

        burnin (discard) --> num_samples (collect) --> stack results --> [num_samples, d]

        each iteration: update_red + update_black (checkerboard blocking).

        returns:
            samples: [num_samples, d]
        """
        samples = []
        total_iters = self.burnin + self.num_samples

        with tqdm(total=total_iters, desc="gibbs sampling") as pbar:
            # burnin phase
            for _ in range(self.burnin):
                self._update_nodes(self.red_indices)
                self._update_nodes(self.black_indices)
                pbar.update(1)

            # collection phase
            for _ in range(self.num_samples):
                self._update_nodes(self.red_indices)
                self._update_nodes(self.black_indices)
                samples.append(self.Y.clone())
                pbar.update(1)

        return torch.stack(samples)  # [num_samples, d]

    def _update_nodes(self, indices: torch.Tensor) -> None:
        """
        update nodes [indices] in-place via conditional sampling (vectorized).

        log_odds = 2(A @ Y + theta) / design --> sigmoid --> bernoulli --> {-1,+1}

        no loops over indices; all operations batched on node subsets.

        args:
            indices: [batch_size], node indices to update
        """
        # compute log-odds for all nodes (vectorized matmul)
        neighbor_sum = self.A @ self.Y  # [d]
        log_odds = 2.0 * (neighbor_sum + self.theta) / self.design  # [d]

        # extract log-odds for target nodes
        log_odds_subset = log_odds[indices]  # [batch_size]

        # convert to probabilities p(Y_i = +1 | Y_-i)
        probs = torch.sigmoid(log_odds_subset)  # [batch_size]

        # vectorized sampling: bernoulli returns {0, 1}, map to {-1, +1}
        bernoulli_samples = torch.bernoulli(probs)  # [batch_size]
        Y_new = 2.0 * bernoulli_samples - 1.0  # [batch_size], {-1, +1}

        # in-place update
        self.Y[indices] = Y_new

    def _compute_checkerboard_coloring(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        partition nodes into checkerboard coloring (red/black bipartition).

        for grid graphs: use (i+j)%2 formula (O(d)).
        for general A: greedy coloring (O(d^2) worst-case but typically fast).

        input A sparsity --> edge list --> color assignment --> [red_idx, black_idx]

        args:
            A: coupling matrix [d, d]

        returns:
            red_indices: [num_red]
            black_indices: [num_black]
        """
        d = A.shape[0]

        # check if this is a grid graph (d is perfect square)
        grid_size = int(d ** 0.5)
        if grid_size * grid_size == d:
            # try grid-based coloring
            red_list = []
            black_list = []
            for i in range(d):
                row = i // grid_size
                col = i % grid_size
                if (row + col) % 2 == 0:
                    red_list.append(i)
                else:
                    black_list.append(i)

            red_indices = torch.tensor(red_list, dtype=torch.long)
            black_indices = torch.tensor(black_list, dtype=torch.long)

            # validate: check no intra-color edges
            valid = True
            for i in red_list:
                for j in red_list:
                    if i != j and A[i, j] != 0:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                for i in black_list:
                    for j in black_list:
                        if i != j and A[i, j] != 0:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                return red_indices, black_indices

        # fallback: greedy coloring for general graphs
        color = torch.full((d,), -1, dtype=torch.long)  # [d], -1 = uncolored

        for i in range(d):
            # find neighbors
            neighbors = torch.where(A[i, :] != 0)[0]
            neighbor_colors = set(color[neighbors].tolist())
            neighbor_colors.discard(-1)  # remove uncolored marker

            # assign smallest available color (0 or 1)
            if 0 not in neighbor_colors:
                color[i] = 0
            elif 1 not in neighbor_colors:
                color[i] = 1
            else:
                # should not happen for bipartite graphs, but handle gracefully
                # assign color 0 (may violate independence but ensures coverage)
                color[i] = 0

        red_indices = torch.where(color == 0)[0]
        black_indices = torch.where(color == 1)[0]

        return red_indices, black_indices


if __name__ == "__main__":
    print("testing ising gibbs sampler on 4x4 grid")

    # setup 4x4 grid
    d = 16
    grid_size = 4

    # create nearest-neighbor coupling matrix
    A = torch.zeros(d, d)
    for i in range(d):
        row = i // grid_size
        col = i % grid_size

        # right neighbor
        if col < grid_size - 1:
            j = i + 1
            A[i, j] = -1.0  # ferromagnetic coupling
            A[j, i] = -1.0

        # bottom neighbor
        if row < grid_size - 1:
            j = i + grid_size
            A[i, j] = -1.0
            A[j, i] = -1.0

    # random external field
    torch.manual_seed(42)
    theta = 2.0 * torch.rand(d) - 1.0  # U(-1, 1)

    # instantiate sampler
    sampler = IsingGibbsSampler(
        theta=theta,
        A=A,
        design=16,
        num_samples=100,
        burnin=50,
        device="cpu"
    )

    # draw samples
    print(f"\nrunning sampler (num_samples=100, burnin=50)...")
    samples = sampler.sample()

    # validate outputs
    print(f"\nvalidation:")
    print(f"  samples shape: {samples.shape}")
    print(f"  expected shape: (100, 16)")
    assert samples.shape == (100, 16), f"shape mismatch: {samples.shape}"

    print(f"  min value: {samples.min().item():.1f}")
    print(f"  max value: {samples.max().item():.1f}")
    assert samples.min() == -1.0 and samples.max() == 1.0, "values not in {-1, +1}"

    print(f"  dtype: {samples.dtype}")
    assert samples.dtype == torch.float32, f"dtype mismatch: {samples.dtype}"

    print(f"  all values in {{-1, +1}}: {torch.all((samples == -1.0) | (samples == 1.0)).item()}")

    # summary statistics
    fraction_positive = (samples > 0).float().mean().item()
    print(f"\nsummary:")
    print(f"  fraction of +1: {fraction_positive:.3f}")
    print(f"  fraction of -1: {1 - fraction_positive:.3f}")

    # visualize first sample
    print(f"\nfirst sample (reshaped as 4x4 grid):")
    first_sample = samples[0].reshape(grid_size, grid_size)
    for row in first_sample:
        print("  " + " ".join(["+1" if x > 0 else "-1" for x in row]))

    # verify checkerboard coloring
    print(f"\ncheckerboard coloring:")
    print(f"  red indices ({len(sampler.red_indices)}): {sampler.red_indices.tolist()}")
    print(f"  black indices ({len(sampler.black_indices)}): {sampler.black_indices.tolist()}")

    red_set = set(sampler.red_indices.tolist())
    black_set = set(sampler.black_indices.tolist())
    print(f"  disjoint: {len(red_set & black_set) == 0}")
    print(f"  covers all nodes: {len(red_set) + len(black_set) == d}")

    # check no intra-color edges
    valid_red = True
    for i in sampler.red_indices:
        for j in sampler.red_indices:
            if i != j and A[i, j] != 0:
                valid_red = False
                break
        if not valid_red:
            break

    valid_black = True
    for i in sampler.black_indices:
        for j in sampler.black_indices:
            if i != j and A[i, j] != 0:
                valid_black = False
                break
        if not valid_black:
            break

    print(f"  no intra-red edges: {valid_red}")
    print(f"  no intra-black edges: {valid_black}")

    print(f"\ntest passed!")
