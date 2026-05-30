"""
CPU core allocation registry per method.

Core values are heuristic; validate post-hoc via wall-clock profiling on
target hardware. Expand via PR and new test case when adding methods.
"""

CORES_REGISTRY: dict[str, int] = {
    # Tabular
    "TabularPluginDRE": 1,
    "SmoothedTabularPluginDRE": 1,
    # Fast continuous
    "TSM": 2,
    "BDRE": 2,
    "MDRE": 2,
    # Slow continuous
    "CTSM": 4,
    "VFM": 4,
    "VFMOrthros": 4,
    "FMDRE": 4,
    "FMDRE_S2": 4,
    "TriangularTSM": 4,
    "TriangularTSM_fix": 4,
    "TriangularFMDRE": 4,
    "TriangularMDRE": 4,
    "MultiHeadTriangularTDRE": 4,
    "MultiHeadTDRE": 4,  # same multi-head classifier/epoch budget as the triangular variant
    "TriangularCTSM_V1": 4,
    "TriangularCTSM_V2": 4,
    "TriangularCTSM_V3": 4,
    "TriangularVFM_V1": 4,
    "TriangularVFM_V2": 4,
    "TriangularVFM_V3": 4,
}


def get_cores_for_method(method: str, overrides: dict[str, int] | None = None) -> int:
    """
    Get CPU core requirement for a method, respecting study-level overrides.

    lookup rules:
    1. if overrides is not None and method in overrides: return overrides[method]
    2. else if method in CORES_REGISTRY: return CORES_REGISTRY[method]
    3. else: raise KeyError with method name

    Args:
        method: method name (must exist in CORES_REGISTRY or overrides).
        overrides: optional dict[method_name -> cores] to override registry values.

    Returns:
        int: number of cores required.

    Raises:
        KeyError: if method not found in registry and not in overrides.
    """
    if overrides is not None and method in overrides:
        return overrides[method]
    if method in CORES_REGISTRY:
        return CORES_REGISTRY[method]
    raise KeyError(f"method '{method}' not found in CORES_REGISTRY and not in overrides")
