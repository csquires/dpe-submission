"""alpha-axis grid for the pendulum trajectory KL grid.

shared by step1_create_data, build_traj_alpha, and assemble_traj_grid so the
sequential, parallel-builder, and assembler paths see the exact same alphas
and therefore agree on cfg_hash + cache layout.
"""
from typing import Any, Dict

import numpy as np


def make_alphas(traj_cfg: Dict[str, Any]) -> np.ndarray:
    """alpha grid for traj_kl_grid honoring traj_cfg.alpha_grid.

    kinds:
      "linspace" (default): np.linspace(0, 1, G_alpha)
      "logspace": np.logspace(alpha_log_min, 0, G_alpha) with alpha_log_min < 0;
                  endpoint at alpha=1, dense near alpha=0 to resolve small KL1.
    """
    G = int(traj_cfg["G_alpha"])
    kind = str(traj_cfg.get("alpha_grid", "linspace"))
    if kind == "linspace":
        return np.linspace(0.0, 1.0, G)
    if kind == "logspace":
        log_min = float(traj_cfg.get("alpha_log_min", -4))
        if log_min >= 0:
            raise ValueError(f"alpha_log_min must be negative, got {log_min}")
        return np.logspace(log_min, 0.0, G)
    raise ValueError(f"unknown alpha_grid kind: {kind!r}")
