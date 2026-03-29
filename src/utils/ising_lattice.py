import torch


def build_ising_adjacency(grid_size: int, device: str = "cpu", sparse: bool = True) -> torch.Tensor:
    """
    build adjacency matrix for 2d ising lattice with periodic boundary conditions.

    uses 4-connectivity (up/down/left/right neighbors) on a torus topology. each
    node has exactly 4 neighbors due to periodic wrapping at boundaries.

    algorithm:
    - map 2d grid positions (row, col) to 1d node indices: node_idx = row * grid_size + col
    - for each node, compute 4 neighbors using modular arithmetic for periodic boundaries
    - collect edges in upper triangular form to avoid duplicates
    - construct symmetric adjacency matrix (both (i,j) and (j,i) stored)

    args:
        grid_size: linear dimension of grid (total nodes = grid_size^2)
        device: pytorch device ("cpu" or "cuda")
        sparse: if true, return sparse coo tensor; else dense tensor

    returns:
        adjacency matrix of shape [n, n] where n = grid_size^2
        - symmetric: A[i,j] == A[j,i]
        - zero diagonal: A[i,i] == 0
        - each node has degree 4
        - unweighted: all edge weights are 1.0
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    n = grid_size * grid_size
    edge_list = []

    for node_i in range(n):
        row_i = node_i // grid_size
        col_i = node_i % grid_size

        neighbors = [
            ((row_i - 1) % grid_size, col_i),
            ((row_i + 1) % grid_size, col_i),
            (row_i, (col_i - 1) % grid_size),
            (row_i, (col_i + 1) % grid_size),
        ]

        for row_j, col_j in neighbors:
            node_j = row_j * grid_size + col_j
            if node_j > node_i:
                edge_list.append((node_i, node_j))

    if sparse:
        rows = []
        cols = []
        for i, j in edge_list:
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

        indices = torch.LongTensor([rows, cols])
        values = torch.ones(len(rows), dtype=torch.float)

        return torch.sparse_coo_tensor(indices, values, (n, n), device=device)
    else:
        adj = torch.zeros(n, n, device=device)
        for i, j in edge_list:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        return adj


if __name__ == "__main__":
    print("=" * 60)
    print("test 1: small grid inspection (3x3)")
    print("=" * 60)

    adj = build_ising_adjacency(grid_size=3, device="cpu", sparse=True)
    adj_dense = adj.to_dense()

    assert adj_dense.shape == (9, 9), "wrong shape"
    assert torch.allclose(torch.diag(adj_dense), torch.zeros(9)), "diagonal not zero"
    assert torch.allclose(adj_dense, adj_dense.T), "not symmetric"
    assert adj_dense[0].sum() == 4.0, "node degree should be 4"

    print(f"shape: {adj_dense.shape}")
    print(f"diagonal all zero: {torch.allclose(torch.diag(adj_dense), torch.zeros(9))}")
    print(f"symmetric: {torch.allclose(adj_dense, adj_dense.T)}")
    print(f"node 0 degree: {adj_dense[0].sum().item()}")
    print("✓ test 1 passed\n")

    print("=" * 60)
    print("test 2: correct sparsity (10x10)")
    print("=" * 60)

    adj = build_ising_adjacency(grid_size=10, device="cpu", sparse=True)
    n = 100
    expected_nnz = 4 * n
    actual_nnz = adj._nnz()

    assert actual_nnz == expected_nnz, f"expected {expected_nnz} edges, got {actual_nnz}"

    adj_dense = adj.to_dense()
    row_sums = adj_dense.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(100) * 4), "all nodes should have degree 4"

    print(f"shape: {adj.shape}")
    print(f"non-zeros: {actual_nnz} (expected: {expected_nnz})")
    print(f"all degrees are 4: {torch.allclose(row_sums, torch.ones(100) * 4)}")
    print("✓ test 2 passed\n")

    print("=" * 60)
    print("test 3: dense vs sparse equivalence (5x5)")
    print("=" * 60)

    grid_sz = 5
    adj_sparse = build_ising_adjacency(grid_size=grid_sz, sparse=True, device="cpu")
    adj_dense_explicit = build_ising_adjacency(grid_size=grid_sz, sparse=False, device="cpu")

    adj_sparse_to_dense = adj_sparse.to_dense()
    assert torch.allclose(adj_sparse_to_dense, adj_dense_explicit), \
        "sparse and dense versions should be identical"

    print(f"sparse shape: {adj_sparse.shape}")
    print(f"dense shape: {adj_dense_explicit.shape}")
    print(f"equivalent: {torch.allclose(adj_sparse_to_dense, adj_dense_explicit)}")
    print("✓ test 3 passed\n")

    print("=" * 60)
    print("test 4: periodic boundary conditions (4x4)")
    print("=" * 60)

    adj = build_ising_adjacency(grid_size=4, device="cpu", sparse=True)
    adj_dense = adj.to_dense()

    expected_neighbors_0 = {1, 3, 4, 12}
    actual_neighbors_0 = set(torch.nonzero(adj_dense[0]).squeeze().tolist())
    assert actual_neighbors_0 == expected_neighbors_0, \
        f"node 0 neighbors {actual_neighbors_0} != {expected_neighbors_0}"

    expected_neighbors_15 = {3, 11, 12, 14}
    actual_neighbors_15 = set(torch.nonzero(adj_dense[15]).squeeze().tolist())
    assert actual_neighbors_15 == expected_neighbors_15, \
        f"node 15 neighbors {actual_neighbors_15} != {expected_neighbors_15}"

    print(f"node 0 neighbors: {actual_neighbors_0} (expected: {expected_neighbors_0})")
    print(f"node 15 neighbors: {actual_neighbors_15} (expected: {expected_neighbors_15})")
    print("✓ test 4 passed\n")

    print("=" * 60)
    print("test 5: device compatibility")
    print("=" * 60)

    adj_cpu = build_ising_adjacency(grid_size=5, device="cpu", sparse=True)
    assert adj_cpu.device.type == "cpu", "should be on cpu"
    print(f"cpu device: {adj_cpu.device}")

    if torch.cuda.is_available():
        adj_cuda = build_ising_adjacency(grid_size=5, device="cuda", sparse=True)
        assert adj_cuda.device.type == "cuda", "should be on cuda"
        print(f"cuda device: {adj_cuda.device}")
        print("✓ test 5 passed (cuda available)\n")
    else:
        print("cuda not available, skipping cuda test")
        print("✓ test 5 passed (cpu only)\n")

    print("=" * 60)
    print("all tests passed!")
    print("=" * 60)
