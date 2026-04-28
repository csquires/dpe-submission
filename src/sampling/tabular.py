import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


def sample_occupancy(
    d: np.ndarray,
    n: int,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    sample (s, a) pairs from categorical distribution d via multinomial.

    d [|S|, |A|] is the occupancy tensor (e.g. d^O or d^E).
    returns cpu int64 tensors s_idx[n], a_idx[n] with valid indices.

    process:
        flatten d to length |S|*|A|
        assert non-negative and renormalize if drift (sum to 1 within tolerance)
        convert to torch.float64 on cpu; assert d_t.is_cpu == True
        draw idx_flat ~ multinomial(d_t.flatten(), n, replacement=true, generator=g)
        s_idx = idx_flat // n_actions; a_idx = idx_flat % n_actions
        return (s_idx.to(torch.int64), a_idx.to(torch.int64)) both on cpu

    edge cases:
        - d has zero-mass cells: multinomial skips them; OK.
        - d sums to x < 1 or x > 1 due to numerical drift: renormalize inline.
          renormalization: d_flat = d_flat / d_flat.sum() before conversion to torch.
        - generator is not None but passed generator.device.type != "cpu":
          raise AssertionError("generator must be on CPU").
    """
    if generator is not None:
        assert generator.device.type == "cpu", "generator must be on CPU"

    # flatten and renormalize
    d_flat = d.flatten()
    d_sum = d_flat.sum()
    if d_sum < 1.0 or d_sum > 1.0:
        d_flat = d_flat / d_sum

    # convert to torch on cpu
    d_t = torch.from_numpy(d_flat).to(torch.float64)
    assert d_t.is_cpu, "tensor must be on CPU"

    # multinomial sampling
    idx_flat = torch.multinomial(d_t, n, replacement=True, generator=generator)

    # recover state and action indices
    n_actions = d.shape[1]
    s_idx = (idx_flat // n_actions).to(torch.int64)
    a_idx = (idx_flat % n_actions).to(torch.int64)

    return s_idx, a_idx


def grid_coord_angle(
    s_idx: torch.Tensor,
    a_idx: torch.Tensor,
    L: int,
    n_actions: int,
    embed_dim: int = 6,
) -> torch.Tensor:
    """
    embed (s, a) into R^embed_dim via normalized grid coords + action angle.

    phi(s, a) = [row/L', col/L', cos(theta), sin(theta), 0, ..., 0]
    where row, col in [0, L-1], theta in [0, 2*pi), padding to embed_dim.

    process:
        row = (s_idx // L).float() / max(L - 1, 1)  # [N]
        col = (s_idx % L).float() / max(L - 1, 1)   # [N]
        theta = 2 * pi * a_idx.float() / n_actions  # [N]
        core = stack([row, col, cos(theta), sin(theta)], dim=-1)  # [N, 4]
        if embed_dim > 4: pad on the right with zeros to [N, embed_dim]
        return core.to(torch.float32) on cpu

    notes:
        - normalization: row, col in [0, 1] for L >= 2; if L == 1, use 0.
        - pi constant: use torch.pi (native, avoids import overhead).
        - no learnable parameters; pure embedding.
    """
    # grid coordinates
    denom = max(L - 1, 1)
    row = (s_idx // L).float() / denom  # [N]
    col = (s_idx % L).float() / denom   # [N]

    # action angle
    theta = 2 * torch.pi * a_idx.float() / n_actions  # [N]

    # core features [N, 4]
    core = torch.stack([row, col, torch.cos(theta), torch.sin(theta)], dim=-1)

    # pad to embed_dim if needed
    if embed_dim > 4:
        padding = torch.zeros(core.shape[0], embed_dim - 4, dtype=torch.float32)
        core = torch.cat([core, padding], dim=-1)

    return core.to(torch.float32)


def encode_sa(
    s_idx: torch.Tensor,
    a_idx: torch.Tensor,
    encoding_cfg: dict,
) -> torch.Tensor:
    """
    encode (s, a) pairs into continuous features via dispatch on encoding_cfg["type"].

    returns phi[N, d] float32 on cpu, where d depends on encoding type.

    dispatch logic (four arms):

    type == "onehot_joint":
        joint_idx = s_idx * encoding_cfg["n_actions"] + a_idx  # [N]
        return F.one_hot(joint_idx, encoding_cfg["n_states"] * encoding_cfg["n_actions"]).float()
        output shape: [N, |S|*|A|]

    type == "onehot_concat":
        s_oh = F.one_hot(s_idx, encoding_cfg["n_states"]).float()  # [N, |S|]
        a_oh = F.one_hot(a_idx, encoding_cfg["n_actions"]).float()  # [N, |A|]
        return torch.cat([s_oh, a_oh], dim=-1)
        output shape: [N, |S| + |A|]

    type == "gaussian_blob":
        phi = grid_coord_angle(s_idx, a_idx, encoding_cfg["L"], encoding_cfg["n_actions"], encoding_cfg["embed_dim"])
        noise = torch.randn(s_idx.shape[0], encoding_cfg["embed_dim"]) * encoding_cfg["sigma"]
        return phi + noise
        output shape: [N, embed_dim]

    type == "flow_pushforward":
        phi = grid_coord_angle(s_idx, a_idx, encoding_cfg["L"], encoding_cfg["n_actions"], encoding_cfg["embed_dim"])
        z = phi + encoding_cfg["sigma"] * torch.randn(s_idx.shape[0], encoding_cfg["embed_dim"])
        flow_module = encoding_cfg["flow_module"]  # pre-instantiated FrozenFlow
        x, _ = flow_module(z)  # unpack (samples, logdetJ); discard logdetJ here
        return x
        output shape: [N, embed_dim]

    required config keys (all types):
        type: str in {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"}
        n_states: int
        n_actions: int
        L: int (gridworld side length)

    required config keys (gaussian_blob, flow_pushforward):
        embed_dim: int, default 6
        embed_fn: str = "grid_coord_angle" (validator; only value currently supported)
        sigma: float

    required config keys (flow_pushforward only):
        flow_module: FrozenFlow instance (passed in pre-instantiated, not constructed here)

    edge cases:
        - flow_pushforward without flow_module in cfg: raise KeyError("flow_module").
        - unknown type: raise ValueError(f"unknown encoding type: {encoding_cfg['type']}").
    """
    enc_type = encoding_cfg["type"]

    if enc_type == "onehot_joint":
        joint_idx = s_idx * encoding_cfg["n_actions"] + a_idx
        return F.one_hot(joint_idx, encoding_cfg["n_states"] * encoding_cfg["n_actions"]).float()

    elif enc_type == "onehot_concat":
        s_oh = F.one_hot(s_idx, encoding_cfg["n_states"]).float()
        a_oh = F.one_hot(a_idx, encoding_cfg["n_actions"]).float()
        return torch.cat([s_oh, a_oh], dim=-1)

    elif enc_type == "gaussian_blob":
        phi = grid_coord_angle(
            s_idx, a_idx, encoding_cfg["L"], encoding_cfg["n_actions"], encoding_cfg["embed_dim"]
        )
        noise = torch.randn(s_idx.shape[0], encoding_cfg["embed_dim"]) * encoding_cfg["sigma"]
        return phi + noise

    elif enc_type == "flow_pushforward":
        phi = grid_coord_angle(
            s_idx, a_idx, encoding_cfg["L"], encoding_cfg["n_actions"], encoding_cfg["embed_dim"]
        )
        z = phi + encoding_cfg["sigma"] * torch.randn(s_idx.shape[0], encoding_cfg["embed_dim"])
        flow_module = encoding_cfg["flow_module"]
        x, _ = flow_module(z)
        return x

    else:
        raise ValueError(f"unknown encoding type: {enc_type}")


def pointwise_discrete_ldr(
    s_idx: torch.Tensor,
    a_idx: torch.Tensor,
    d_O: np.ndarray,
    d_E: np.ndarray,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    pointwise discrete log-density-ratio via direct lookup.

    log r(s, a) = log d_O[s, a] - log d_E[s, a]

    process:
        d_O_t = torch.from_numpy(d_O).to(torch.float32)  # [|S|, |A|]
        d_E_t = torch.from_numpy(d_E).to(torch.float32)  # [|S|, |A|]
        gathered_O = d_O_t[s_idx, a_idx] + eps  # [N], safeguard against log(0)
        gathered_E = d_E_t[s_idx, a_idx] + eps  # [N]
        return torch.log(gathered_O) - torch.log(gathered_E)  # [N], float32

    edge cases:
        - s_idx or a_idx out of range: pytorch indexing raises IndexError (acceptable).
        - d_O[s, a] + eps == 0: impossible (eps > 0), but log(eps) is safe.
        - d_O[s, a] == 0: replaced by eps; log(eps) is large negative.
    """
    d_O_t = torch.from_numpy(d_O).to(torch.float32)
    d_E_t = torch.from_numpy(d_E).to(torch.float32)

    gathered_O = d_O_t[s_idx, a_idx] + eps  # [N]
    gathered_E = d_E_t[s_idx, a_idx] + eps  # [N]

    return torch.log(gathered_O) - torch.log(gathered_E)  # [N]


def pointwise_smoothed_ldr(
    x: torch.Tensor,
    encoding_cfg: dict,
    d_O: np.ndarray,
    d_E: np.ndarray,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    pointwise smoothed log-density-ratio via analytic kernel (Gaussian or flow-pushed).

    log r_tilde(x) = log p_tilde_O(x) - log p_tilde_E(x)
    where p_tilde_O(x) = sum_{s,a} d_O[s,a] * k(x | phi(s,a), sigma)

    dispatch on encoding_cfg["type"]:

    ===== type == "onehot_joint" or "onehot_concat" =====
    smoothed equals discrete (delta kernel on one-hot vertex set).
    recover (s, a) from x and call pointwise_discrete_ldr by argmax.

    for onehot_joint:
        joint_idx = torch.argmax(x, dim=-1)  # [N]
        s_idx = joint_idx // encoding_cfg["n_actions"]
        a_idx = joint_idx % encoding_cfg["n_actions"]
        return pointwise_discrete_ldr(s_idx, a_idx, d_O, d_E, eps)

    for onehot_concat:
        s_idx = torch.argmax(x[:, :encoding_cfg["n_states"]], dim=-1)
        a_idx = torch.argmax(x[:, encoding_cfg["n_states"]:], dim=-1)
        return pointwise_discrete_ldr(s_idx, a_idx, d_O, d_E, eps)

    ===== type == "gaussian_blob" or "flow_pushforward" =====
    build [|S|*|A|, embed_dim] grid of all (s, a) pairs via grid_coord_angle.
    for each x[i], compute log_components[i, j] over all j in {0,...,|S|*|A|-1}.

    procedure:
        |S| = encoding_cfg["n_states"]
        |A| = encoding_cfg["n_actions"]
        L = encoding_cfg["L"]
        sigma = encoding_cfg["sigma"]
        embed_dim = encoding_cfg["embed_dim"]

        # construct all (s, a) indices
        s_grid, a_grid = torch.meshgrid(torch.arange(|S|), torch.arange(|A|), indexing="ij")  # each [|S|, |A|]
        s_flat = s_grid.flatten()  # [|S|*|A|]
        a_flat = a_grid.flatten()  # [|S|*|A|]

        # compute phi(s, a) for all pairs
        phi_grid = grid_coord_angle(s_flat, a_flat, L, |A|, embed_dim)  # [|S|*|A|, embed_dim]

        # flatten d_O, d_E
        d_O_flat = torch.from_numpy(d_O).float().flatten()  # [|S|*|A|]
        d_E_flat = torch.from_numpy(d_E).float().flatten()  # [|S|*|A|]

    for gaussian_blob:
        # log_components[i, j] = log(d_O[j]) + log N(x[i] | phi_grid[j], sigma^2 * I)
        #                       = log(d_O[j]) - 0.5 * embed_dim * log(2*pi*sigma^2) - ||x[i] - phi_grid[j]||^2 / (2*sigma^2)

        sq_dist = torch.cdist(x, phi_grid) ** 2  # [N, |S|*|A|]
        const_term = -0.5 * embed_dim * torch.log(2 * torch.pi * sigma ** 2)
        log_components_O = torch.log(d_O_flat + eps) + const_term - sq_dist / (2 * sigma ** 2)  # [N, |S|*|A|]
        log_components_E = torch.log(d_E_flat + eps) + const_term - sq_dist / (2 * sigma ** 2)  # [N, |S|*|A|]

        log_p_tilde_O = torch.logsumexp(log_components_O, dim=1)  # [N]
        log_p_tilde_E = torch.logsumexp(log_components_E, dim=1)  # [N]

        return log_p_tilde_O - log_p_tilde_E  # [N], float32

    for flow_pushforward:
        # z = flow.inverse(x) with log|det J_inv| return
        # log_components[i, j] = log(d_O[j]) + log N(z[i] | phi_grid[j], sigma^2 * I) + log|det J_inv(x[i])|
        # note: log|det J_inv| is the same constant for all components j, so it cancels in the difference.
        #       skip computing it; only use the Gaussian term.

        z, log_det_inv_J = encoding_cfg["flow_module"].inverse(x)  # z[N, embed_dim], log_det_inv_J[N]

        sq_dist = torch.cdist(z, phi_grid) ** 2  # [N, |S|*|A|]
        const_term = -0.5 * embed_dim * torch.log(2 * torch.pi * sigma ** 2)
        log_components_O = torch.log(d_O_flat + eps) + const_term - sq_dist / (2 * sigma ** 2)  # [N, |S|*|A|]
        log_components_E = torch.log(d_E_flat + eps) + const_term - sq_dist / (2 * sigma ** 2)  # [N, |S|*|A|]

        log_p_tilde_O = torch.logsumexp(log_components_O, dim=1)  # [N]
        log_p_tilde_E = torch.logsumexp(log_components_E, dim=1)  # [N]

        return log_p_tilde_O - log_p_tilde_E  # [N], float32

    memory note:
        log_components is [N, |S|*|A|].
        for typical N=5000, |S|*|A|=256: 5MB (OK).
        for N=40000, |S|*|A|=1600: 256MB (possible bottleneck).
        if memory exhausted, chunk over N (e.g. batches of 5000) and concatenate results.

    required config keys (all smoothed types):
        type: str
        n_states: int
        n_actions: int
        L: int

    required config keys (gaussian_blob, flow_pushforward):
        embed_dim: int
        sigma: float

    required config keys (flow_pushforward only):
        flow_module: FrozenFlow instance

    edge cases:
        - onehot type and x is not one-hot (e.g. due to encoding error):
          argmax still recovers an index, but mapping is undefined. OK (catch in tests).
        - flow_pushforward without flow_module: raise KeyError("flow_module").
        - unknown type: raise ValueError.
    """
    enc_type = encoding_cfg["type"]

    if enc_type == "onehot_joint":
        joint_idx = torch.argmax(x, dim=-1)  # [N]
        s_idx = joint_idx // encoding_cfg["n_actions"]
        a_idx = joint_idx % encoding_cfg["n_actions"]
        return pointwise_discrete_ldr(s_idx, a_idx, d_O, d_E, eps)

    elif enc_type == "onehot_concat":
        s_idx = torch.argmax(x[:, :encoding_cfg["n_states"]], dim=-1)
        a_idx = torch.argmax(x[:, encoding_cfg["n_states"]:], dim=-1)
        return pointwise_discrete_ldr(s_idx, a_idx, d_O, d_E, eps)

    elif enc_type == "gaussian_blob" or enc_type == "flow_pushforward":
        # construct grid
        n_states = encoding_cfg["n_states"]
        n_actions = encoding_cfg["n_actions"]
        L = encoding_cfg["L"]
        sigma = encoding_cfg["sigma"]
        embed_dim = encoding_cfg["embed_dim"]

        s_grid, a_grid = torch.meshgrid(
            torch.arange(n_states), torch.arange(n_actions), indexing="ij"
        )
        s_flat = s_grid.flatten()  # [|S|*|A|]
        a_flat = a_grid.flatten()  # [|S|*|A|]

        phi_grid = grid_coord_angle(s_flat, a_flat, L, n_actions, embed_dim)  # [|S|*|A|, embed_dim]
        # align device with xs so cdist / arithmetic does not raise
        phi_grid = phi_grid.to(x.device)

        d_O_flat = torch.from_numpy(d_O).float().flatten().to(x.device)  # [|S|*|A|]
        d_E_flat = torch.from_numpy(d_E).float().flatten().to(x.device)  # [|S|*|A|]

        # the gaussian normalization constant -0.5 * embed_dim * log(2 pi sigma^2)
        # is identical between the O- and E-mixtures, so it cancels in the difference.
        # we omit it. similarly, in the flow branch log|det J^{-1}(x)| is the same
        # constant offset for both measures and cancels.
        if enc_type == "gaussian_blob":
            sq_dist = torch.cdist(x, phi_grid) ** 2  # [N, |S|*|A|]
        else:  # flow_pushforward
            z, _ = encoding_cfg["flow_module"].inverse(x)
            sq_dist = torch.cdist(z, phi_grid) ** 2  # [N, |S|*|A|]

        log_components_O = torch.log(d_O_flat + eps) - sq_dist / (2 * sigma ** 2)
        log_components_E = torch.log(d_E_flat + eps) - sq_dist / (2 * sigma ** 2)
        log_p_tilde_O = torch.logsumexp(log_components_O, dim=1)  # [N]
        log_p_tilde_E = torch.logsumexp(log_components_E, dim=1)  # [N]
        return log_p_tilde_O - log_p_tilde_E

    else:
        raise ValueError(f"unknown encoding type: {enc_type}")
