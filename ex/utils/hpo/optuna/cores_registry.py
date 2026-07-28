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
    # dokls ablation cls loss variants (cpu; not in NEEDS_GPU)
    "BDRE_NWJ": 2,
    "BDRE_DV": 2,
    "MHT_NWJ": 4,
    "MHT_DV": 4,
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
    # peak-campaign fallback: strip `_peak` suffix and inherit the base
    # method's cores allocation. peak variants share their base's compute
    # shape (same builder, same per-step cost), so the base value is right.
    if method.endswith("_peak"):
        base = method[:-len("_peak")]
        if overrides is not None and base in overrides:
            return overrides[base]
        if base in CORES_REGISTRY:
            return CORES_REGISTRY[base]
    raise KeyError(f"method '{method}' not found in CORES_REGISTRY and not in overrides")


# methods whose continuous-time ode integration / score / flow-matching
# memory cost makes them OOM on cpu/array lanes at batch_size=1024 (added to
# the peak campaign's widened search space). gating these to gpu lanes only
# stops mnist trial FAIL rates of 70-95% observed for the affected families.
# classifier-style methods (BDRE/MDRE/MHTDRE/TriangularMDRE etc.) fit on
# cpu and stay unrestricted.
NEEDS_GPU: set[str] = {
    "TSM", "CTSM", "VFM", "VFMOrthros",
    "FMDRE", "FMDRE_S2",
    "TriangularTSM", "TriangularTSM_fix", "TriangularFMDRE",
    "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
    "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3",
}


def needs_gpu(method: str) -> bool:
    """return True if method must run on a gpu lane (gpus >= 1).

    strips ``_peak`` suffix so peak campaign variants inherit the base
    method's gating. methods not in ``NEEDS_GPU`` return False (unrestricted).
    """
    if method in NEEDS_GPU:
        return True
    if method.endswith("_peak"):
        return method[:-len("_peak")] in NEEDS_GPU
    return False
