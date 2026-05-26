"""Core test infrastructure: seed control, hyperparameter defaults, path builders, mc sampling."""
import random
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

from src.waypoints.path_builders import (
    stiff_noise,
    bridge_noise,
    stiff_noise_2d,
    bridge_noise_2d,
    direct_1d,
    bary_1d,
    psb_1d,
    rect_2d,
)
from src.waypoints.dataclass_paths import DirectPath1D, TriangularPath1D, TriangularPath2D


def seed_everything(seed: int) -> None:
    """Set global random seeds for deterministic test execution.

    Controls: torch, numpy, python random, and cuda (defensively).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def default_hp(method: str, **overrides) -> dict:
    """Return minimal valid hyperparameter dict for a flow method.

    Includes all base defaults, method-specific deltas (vertex for triangular v1/v2,
    vertex+t2_max+path_height for v3), and applies overrides.

    Args:
        method: one of TSM, CTSM, CTSM-V1, CTSM-V2, CTSM-V3, FMDRE, FMDRE-S2, BDRE,
                MDRE, MDRE-V1, VFM, VFM-V1, VFM-V2, VFM-V3, TriangularFMDRE,
                TriangularMDRE, TabularpluginDRE, SmoothedTabularPluginDRE,
                MultiHeadTDRE, MultiHeadTriangularTDRE.
        **overrides: dict update applied to base + method-specific dict.

    Returns:
        dict with all base keys, method-specific deltas, and overrides applied.
    """
    # sized so a correct method on a tractable closed-form LDR (gaussian-gaussian)
    # actually converges. tests are method-correctness checks, not HPO-speed checks.
    base = {
        "n_epochs": 2000,
        "batch_size": 256,
        "hidden_dim": 128,
        "n_hidden_layers": 3,
        "lr": 1e-3,
        "sigma": 1.0,
        "integration_steps": 500,
        "eps": 1e-3,
        "test_eps": 1e-3,
        "sched": "stiff",
        "inner_eps": 0.0,
        "gamma_min": 0.05,
        "k": 20,
        "ema_decay": 0.999,
        "grad_clip_norm": 1.0,
        "weight_decay": 0.0,
        "activation": "silu",
        "time_dist": "uniform",
        "reweight": False,
        "cosine_min_factor": 0.0,
        "precond": False,
    }

    # method-specific deltas
    if method in ["CTSM-V1", "CTSM-V2", "VFM-V1", "VFM-V2", "TriangularFMDRE", "TriangularMDRE"]:
        base.update({"vertex": 0.5})
    elif method in ["CTSM-V3", "VFM-V3"]:
        base.update({"vertex": 0.5, "t2_max": 0.8, "path_height": 1.5})

    base.update(overrides)
    return base


def make_direct_path_1d(
    sigma: float = 1.0,
    sched: str = "stiff",
    k: float = 20.0,
    inner_eps: float = 0.0,
    gamma_min: float = 0.05,
    eps: float = 1e-3,
) -> DirectPath1D:
    """Construct a DirectPath1D with the specified schedule.

    Translates flat scalar parameters to a schedule factory call and path builder.
    Sched must be 'stiff' (using k) or 'bridge' (sigma only).

    Returns:
        DirectPath1D from src.waypoints.dataclass_paths.
    """
    if sched == "stiff":
        sched_obj = stiff_noise(k=k, sigma=sigma)
    elif sched == "bridge":
        sched_obj = bridge_noise(sigma=sigma)
    else:
        raise ValueError(f"sched must be 'stiff' or 'bridge', got {sched}")

    return direct_1d(
        sched=sched_obj,
        inner_eps=inner_eps,
        gamma_min=gamma_min,
        eps=eps,
    )


def make_triangular_path_1d(
    sigma: float = 1.0,
    sched: str = "stiff",
    vertex: float = 0.5,
    k: float = 20.0,
    inner_eps: float = 0.0,
    gamma_min: float = 0.05,
    eps: float = 1e-3,
    *,
    kind: str = "bary",
) -> TriangularPath1D:
    """Construct a TriangularPath1D with the specified schedule and weights family.

    Sched must be 'stiff' or 'bridge'. Kind must be 'bary' (barycentric) or 'psb'
    (piecewise-sb).

    Returns:
        TriangularPath1D from src.waypoints.dataclass_paths.
    """
    if sched == "stiff":
        sched_obj = stiff_noise(k=k, sigma=sigma)
    elif sched == "bridge":
        sched_obj = bridge_noise(sigma=sigma)
    else:
        raise ValueError(f"sched must be 'stiff' or 'bridge', got {sched}")

    if kind == "bary":
        return bary_1d(
            sched=sched_obj,
            vertex=vertex,
            inner_eps=inner_eps,
            gamma_min=gamma_min,
            eps=eps,
        )
    elif kind == "psb":
        return psb_1d(
            sched=sched_obj,
            vertex=vertex,
            inner_eps=inner_eps,
            gamma_min=gamma_min,
            eps=eps,
        )
    else:
        raise ValueError(f"kind must be 'bary' or 'psb', got {kind}")


def make_triangular_path_2d(
    sigma: float = 1.0,
    sched: str = "stiff",
    k: float = 20.0,
    t2_max: float = 0.8,
    eps: float = 1e-3,
) -> TriangularPath2D:
    """Construct a TriangularPath2D with the specified schedule.

    Sched must be 'stiff' or 'bridge'. t2_max and path_height are consumed by the
    sampler/curve, not the path geometry itself.

    Returns:
        TriangularPath2D from src.waypoints.dataclass_paths.
    """
    if sched == "stiff":
        sched_obj = stiff_noise_2d(k=k, sigma=sigma)
    elif sched == "bridge":
        sched_obj = bridge_noise_2d(sigma=sigma)
    else:
        raise ValueError(f"sched must be 'stiff' or 'bridge', got {sched}")

    return rect_2d(
        sched=sched_obj,
        inner_eps=0.0,
        gamma_min=0.0,
        eps=eps,
    )


def monte_carlo_target_ratio(
    target_fn: Callable,
    path: DirectPath1D | TriangularPath1D | TriangularPath2D,
    x0: Tensor,
    x1: Tensor,
    tau: Tensor,
    n_samples: int,
    *,
    xstar: Optional[Tensor] = None,
    seed: int = 0,
) -> dict:
    """Draw noise samples and evaluate target functions against diffusion logs.

    For each of n_samples iterations, draws eps ~ N(0, I_D) and calls the target
    function. Accumulates and stacks results. Returns x_tau, target, lambda_t, and
    their ratio.

    Args:
        target_fn: Callable with signature target_fn(path, x0, x1, [xstar,] tau, eps)
                   returning (x_tau, target, lambda_t).
        path: DirectPath1D, TriangularPath1D, or TriangularPath2D.
        x0: Tensor[B, D], starting points.
        x1: Tensor[B, D], ending points.
        tau: Tensor[B, 1], time values.
        n_samples: int, number of noise samples to draw.
        xstar: Tensor[B, D] | None, anchor point (required for triangular paths).
        seed: int = 0, for reproducibility.

    Returns:
        dict with keys:
            x_tau_samples: Tensor[n_samples, B, D]
            target_samples: Tensor[n_samples, B, 1]
            lambda_t_samples: Tensor[n_samples, B, 1]
            ratio_samples: Tensor[n_samples, B, 1] = target / lambda_t
    """
    seed_everything(seed)

    B, D = x0.shape
    is_triangular = xstar is not None

    # accumulate samples
    x_tau_list = []
    target_list = []
    lambda_t_list = []

    for _ in range(n_samples):
        eps = torch.randn(B, D)  # batch B, dims D
        if is_triangular:
            x_tau_i, target_i, lambda_t_i = target_fn(path, x0, x1, xstar, tau, eps)
        else:
            x_tau_i, target_i, lambda_t_i = target_fn(path, x0, x1, tau, eps)
        x_tau_list.append(x_tau_i)
        target_list.append(target_i)
        lambda_t_list.append(lambda_t_i)

    # stack samples: [n_samples, B, ...]
    x_tau_samples = torch.stack(x_tau_list, dim=0)  # [n_samples, B, D]
    target_samples = torch.stack(target_list, dim=0)  # [n_samples, B, 1]
    lambda_t_samples = torch.stack(lambda_t_list, dim=0)  # [n_samples, B, 1]

    # compute ratio with small epsilon to avoid division by zero
    ratio_samples = target_samples / (lambda_t_samples + 1e-10)

    return {
        "x_tau_samples": x_tau_samples.to(dtype=torch.float32, device="cpu"),
        "target_samples": target_samples.to(dtype=torch.float32, device="cpu"),
        "lambda_t_samples": lambda_t_samples.to(dtype=torch.float32, device="cpu"),
        "ratio_samples": ratio_samples.to(dtype=torch.float32, device="cpu"),
    }
