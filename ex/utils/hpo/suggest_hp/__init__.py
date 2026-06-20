"""suggest_hp registry: maps method names to search space definitions and metadata.

single entrypoint for hyperparameter search space definition. each method
is registered as a tuple of (suggest_hp function, METADATA dict).
"""
from collections.abc import Callable
import optuna

from . import bdre
from . import ctsm
from . import fmdre
from . import fmdre_s2
from . import mdre
from . import mh_tdre
from . import mh_triangular_tdre
from . import triangular_fmdre
from . import triangular_mdre
from . import triangular_ctsm
from . import triangular_tsm
from . import triangular_vfm
from . import tabular_plugin_dre
from . import tsm
from . import vfm
from . import vfmorthros

# peak variants — add ONE line per peak module that exists on disk (strict
# "no import without a file" rule per spec 09).
from . import bdre_peak
from . import vfm_peak
from . import ctsm_peak
from . import tsm_peak
from . import triangular_ctsm_peak
from . import triangular_vfm_peak
from . import mdre_peak
from . import triangular_mdre_peak
from . import mh_tdre_peak
from . import mh_triangular_tdre_peak
from . import fmdre_peak
from . import fmdre_s2_peak
from . import triangular_fmdre_peak
from . import triangular_tsm_peak


SUGGEST_HP_REGISTRY: dict[str, tuple[Callable[[optuna.Trial], dict], dict]] = {
    "BDRE": (bdre.suggest_hp, bdre.METADATA),
    "CTSM": (ctsm.suggest_hp, ctsm.METADATA),
    "FMDRE": (fmdre.suggest_hp, fmdre.METADATA),
    "FMDRE_S2": (fmdre_s2.suggest_hp, fmdre_s2.METADATA),
    "MDRE": (mdre.suggest_hp, mdre.METADATA),
    "MultiHeadTDRE": (mh_tdre.suggest_hp, mh_tdre.METADATA),
    "MultiHeadTriangularTDRE": (mh_triangular_tdre.suggest_hp, mh_triangular_tdre.METADATA),
    "TriangularFMDRE": (triangular_fmdre.suggest_hp, triangular_fmdre.METADATA),
    "TriangularMDRE": (triangular_mdre.suggest_hp, triangular_mdre.METADATA),
    "TriangularCTSM_V1": (triangular_ctsm.suggest_hp_v1, triangular_ctsm.METADATA_V1),
    "TriangularCTSM_V2": (triangular_ctsm.suggest_hp_v2, triangular_ctsm.METADATA_V2),
    "TriangularCTSM_V3": (triangular_ctsm.suggest_hp_v3, triangular_ctsm.METADATA_V3),
    "TriangularVFM_V1": (triangular_vfm.suggest_hp_v1, triangular_vfm.METADATA_V1),
    "TriangularVFM_V2": (triangular_vfm.suggest_hp_v2, triangular_vfm.METADATA_V2),
    "TriangularVFM_V3": (triangular_vfm.suggest_hp_v3, triangular_vfm.METADATA_V3),
    "TabularPluginDRE": (tabular_plugin_dre.suggest_hp, tabular_plugin_dre.METADATA),
    "TSM": (tsm.suggest_hp, tsm.METADATA),
    "TriangularTSM": (triangular_tsm.suggest_hp, triangular_tsm.METADATA),
    "VFM": (vfm.suggest_hp, vfm.METADATA),
    "VFMOrthros": (vfmorthros.suggest_hp, vfmorthros.METADATA),

    # peak variants
    "BDRE_peak": (bdre_peak.suggest_hp, bdre_peak.METADATA),
    "VFM_peak": (vfm_peak.suggest_hp, vfm_peak.METADATA),
    "CTSM_peak": (ctsm_peak.suggest_hp, ctsm_peak.METADATA),
    "TSM_peak": (tsm_peak.suggest_hp, tsm_peak.METADATA),
    "TriangularCTSM_V1_peak": (triangular_ctsm_peak.suggest_hp_v1, triangular_ctsm_peak.METADATA_V1),
    "TriangularCTSM_V2_peak": (triangular_ctsm_peak.suggest_hp_v2, triangular_ctsm_peak.METADATA_V2),
    "TriangularCTSM_V3_peak": (triangular_ctsm_peak.suggest_hp_v3, triangular_ctsm_peak.METADATA_V3),
    "TriangularVFM_V1_peak": (triangular_vfm_peak.suggest_hp_v1, triangular_vfm_peak.METADATA_V1),
    "TriangularVFM_V2_peak": (triangular_vfm_peak.suggest_hp_v2, triangular_vfm_peak.METADATA_V2),
    "TriangularVFM_V3_peak": (triangular_vfm_peak.suggest_hp_v3, triangular_vfm_peak.METADATA_V3),
    "MDRE_peak": (mdre_peak.suggest_hp, mdre_peak.METADATA),
    "TriangularMDRE_peak": (triangular_mdre_peak.suggest_hp, triangular_mdre_peak.METADATA),
    "MultiHeadTDRE_peak": (mh_tdre_peak.suggest_hp, mh_tdre_peak.METADATA),
    "MultiHeadTriangularTDRE_peak": (mh_triangular_tdre_peak.suggest_hp, mh_triangular_tdre_peak.METADATA),
    "FMDRE_peak": (fmdre_peak.suggest_hp, fmdre_peak.METADATA),
    "FMDRE_S2_peak": (fmdre_s2_peak.suggest_hp, fmdre_s2_peak.METADATA),
    "TriangularFMDRE_peak": (triangular_fmdre_peak.suggest_hp, triangular_fmdre_peak.METADATA),
    "TriangularTSM_peak": (triangular_tsm_peak.suggest_hp, triangular_tsm_peak.METADATA),
}


def suggest_hp(trial: optuna.Trial, method: str) -> dict:
    """dispatch to registered suggest_hp function for method.

    input: trial (optuna.Trial), method name (str).
    action: lookup (suggest_hp fn, _) from registry; call fn(trial).
    output: dict of hyperparameters.

    raises KeyError if method not in registry; message lists available methods.
    """
    if method not in SUGGEST_HP_REGISTRY:
        available = sorted(SUGGEST_HP_REGISTRY.keys())
        raise KeyError(
            f"method '{method}' not in registry. available: {available}"
        )

    fn, _ = SUGGEST_HP_REGISTRY[method]
    return fn(trial)


def get_metadata(method: str) -> dict:
    """return METADATA dict for method.

    input: method name (str).
    action: lookup (_, metadata) from registry.
    output: metadata dict with keys {cores_per_trial, uses_pruning, requires_pstar, builder}.

    raises KeyError if method not in registry; message lists available methods.
    """
    if method not in SUGGEST_HP_REGISTRY:
        available = sorted(SUGGEST_HP_REGISTRY.keys())
        raise KeyError(
            f"method '{method}' not in registry. available: {available}"
        )

    _, metadata = SUGGEST_HP_REGISTRY[method]
    return metadata


def list_methods() -> list[str]:
    """return list of registered method names.

    output: list[str] of all keys in SUGGEST_HP_REGISTRY.

    use case: ui enumeration, method filter validation.
    """
    return list(SUGGEST_HP_REGISTRY.keys())


def peak_variant(method: str) -> str | None:
    """return peak variant name for method, or None if not registered.

    input: method name (str), e.g. 'BDRE', 'VFM'.
    action: check if f"{method}_peak" exists in registry.
    output: f"{method}_peak" if registered, else None.

    use case: campaign config auto-discovery, parity test setup.
    """
    peak_name = f"{method}_peak"
    return peak_name if peak_name in SUGGEST_HP_REGISTRY else None
