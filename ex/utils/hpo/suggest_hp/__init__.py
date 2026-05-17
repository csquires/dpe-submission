"""suggest_hp registry: maps method names to search space definitions and metadata.

single entrypoint for hyperparameter search space definition. each method
is registered as a tuple of (suggest_hp function, METADATA dict).
"""
from collections.abc import Callable
import optuna

from . import bdre
from . import mh_triangular_tdre
from . import triangular_fmdre
from . import tabular_plugin_dre
from . import vfm
from . import vfmorthros


SUGGEST_HP_REGISTRY: dict[str, tuple[Callable[[optuna.Trial], dict], dict]] = {
    "BDRE": (bdre.suggest_hp, bdre.METADATA),
    "MultiHeadTriangularTDRE": (mh_triangular_tdre.suggest_hp, mh_triangular_tdre.METADATA),
    "TriangularFMDRE": (triangular_fmdre.suggest_hp, triangular_fmdre.METADATA),
    "TabularPluginDRE": (tabular_plugin_dre.suggest_hp, tabular_plugin_dre.METADATA),
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
