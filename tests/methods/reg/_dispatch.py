"""Uniform method registry for flow-based regression methods.

Pure dispatch module (no state). Exposes FLOW_METHODS tuple, PARADIGM_TARGET_FN and
REQUIRES_PSTAR dicts (built at module load time), and get_method(name) function.
All paradigm functions are imported eagerly to catch import errors early.
"""

from typing import Callable

# list all 11 regression flow methods (order fixed per spec).
FLOW_METHODS = (
    "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
    "VFM", "VFMOrthros",
    "FMDRE", "FMDRE_S2", "TSM",
    "TriangularFMDRE", "TriangularMDRE",
)

# import paradigm functions at module level for early error detection.
from src.methods.reg.common._paradigm_funcs import (
    ctsm_regression_target_direct_1d,
    ctsm_regression_target_1d,
    ctsm_regression_target_2d,
    vfm_velocity_target_direct_1d,
)

# mapping of method names to their target paradigm functions.
# verified by reading each method's fit() to see which paradigm function it calls.
# note: FMDRE, FMDRE_S2, TriangularFMDRE, TriangularMDRE, TSM don't call paradigm
# functions; they use flow-matching or classification loss instead.
PARADIGM_TARGET_FN: dict[str, Callable | None] = {
    "CTSM": ctsm_regression_target_direct_1d,  # direct path; 2-source
    "TriangularCTSM_V1": ctsm_regression_target_1d,  # triangular path; 3-source
    "TriangularCTSM_V2": ctsm_regression_target_1d,  # triangular path; 3-source
    "TriangularCTSM_V3": ctsm_regression_target_2d,  # 2d stacked geometry
    "VFM": vfm_velocity_target_direct_1d,  # direct path; 2-source
    "VFMOrthros": vfm_velocity_target_direct_1d,  # direct path; 2-source
    "FMDRE": None,  # flow matching; no paradigm function
    "FMDRE_S2": None,  # flow matching with cfg; no paradigm function
    "TSM": None,  # unconditional score matching; no paradigm function
    "TriangularFMDRE": None,  # triangular flow matching; no paradigm function
    "TriangularMDRE": None,  # multiclass classification; no paradigm function
}

# hardcoded lookup: which methods require third source distribution sample (xstar / pstar).
# verified against METHOD_SPECS in ex.utils.hpo.method_specs.
REQUIRES_PSTAR: dict[str, bool] = {
    "CTSM": False,
    "TriangularCTSM_V1": True,
    "TriangularCTSM_V2": True,
    "TriangularCTSM_V3": True,
    "VFM": False,
    "VFMOrthros": False,
    "FMDRE": False,
    "FMDRE_S2": False,
    "TSM": False,
    "TriangularFMDRE": True,
    "TriangularMDRE": True,
}


def get_method(name: str) -> dict:
    """Return {builder, paradigm_target_fn, requires_pstar} for a flow method.

    Args:
        name: method name (e.g., "CTSM", "TriangularCTSM_V1").

    Returns:
        dict with keys:
          - "builder": Callable[input_dim, device, num_waypoints, **flat_hp] -> Estimator
          - "paradigm_target_fn": Callable or None (paradigm function or None for non-flow methods)
          - "requires_pstar": bool (True if method needs third source distribution)

    Raises:
        ValueError: if name not in FLOW_METHODS.
    """
    if name not in FLOW_METHODS:
        raise ValueError(
            f"unknown flow method '{name}'; valid: {FLOW_METHODS}"
        )

    # lazy import builder to allow harness modules to import _dispatch
    # without all builders being loaded (e.g., TriangularVFM2D dependencies).
    from ex.utils.hpo.builders import BUILDERS_REGISTRY

    builder_name = f"build_{name}"
    if builder_name not in BUILDERS_REGISTRY:
        raise ValueError(
            f"builder '{builder_name}' not in BUILDERS_REGISTRY; "
            f"check ex.utils.hpo.builders for export."
        )

    return {
        "builder": BUILDERS_REGISTRY[builder_name],
        "paradigm_target_fn": PARADIGM_TARGET_FN.get(name),
        "requires_pstar": REQUIRES_PSTAR[name],
    }


def is_triangular(name: str) -> bool:
    """Check if method name indicates a triangular path variant.

    Args:
        name: method name.

    Returns:
        True iff name starts with "Triangular".
    """
    return name.startswith("Triangular")


def paradigm_is_2d(name: str) -> bool:
    """Check if method uses a 2D path paradigm (stacked geometry).

    Args:
        name: method name.

    Returns:
        True iff name is "TriangularCTSM_V3" (only 2D paradigm in v1 scope).
    """
    return name in {"TriangularCTSM_V3"}
