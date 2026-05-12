import torch


def split_for_eval(
    data: dict[str, torch.Tensor],
    *,
    seed: int,
    n_eval_max: int = 256,
    eval_frac: float = 0.2,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """partition paired pstar + true_ldrs into train/eval; pass other keys through.

    input:
      data dict with required keys 'pstar', 'true_ldrs' (paired by index 0)
      and optional 'p0', 'p1', others.

    action:
      compute n_eval = min(n_eval_max, max(1, int(N * eval_frac))).
      generate deterministic permutation from seed on pstar.device.
      split tensors; return train_data (all keys) and eval_data (pstar, true_ldrs only).

    output:
      (train_data, eval_data) tuple; both dicts, same device and dtype as input.

    raises:
      ValueError if true_ldrs.shape[0] != pstar.shape[0].
      KeyError if 'pstar' or 'true_ldrs' absent from data.
    """
    # step 1: extract and validate paired keys.
    pstar = data["pstar"]
    true_ldrs = data["true_ldrs"]
    n = pstar.shape[0]

    if true_ldrs.shape[0] != n:
        raise ValueError(
            f"pstar and true_ldrs size mismatch: {n} vs {true_ldrs.shape[0]}"
        )

    # step 2: compute eval partition size.
    n_eval = min(n_eval_max, max(1, int(n * eval_frac)))

    # step 3: generate seeded permutation on same device as pstar.
    gen = torch.Generator(device=pstar.device).manual_seed(seed)
    perm = torch.randperm(n, generator=gen, device=pstar.device)

    # step 4: split indices.
    eval_indices = perm[:n_eval]
    train_indices = perm[n_eval:]

    # step 5: build train_data with all keys, split on paired keys.
    train_data = {}
    train_data["pstar"] = pstar[train_indices]
    train_data["true_ldrs"] = true_ldrs[train_indices]
    for k, v in data.items():
        if k not in ("pstar", "true_ldrs"):
            train_data[k] = v

    # step 6: build eval_data with only paired keys.
    eval_data = {}
    eval_data["pstar"] = pstar[eval_indices]
    eval_data["true_ldrs"] = true_ldrs[eval_indices]

    # step 7: return tuple.
    return train_data, eval_data
