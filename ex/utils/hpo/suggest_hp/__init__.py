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
