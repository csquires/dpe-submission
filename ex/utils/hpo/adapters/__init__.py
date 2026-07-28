"""
adapter factory: registry and factory functions.

minimal factory pattern. resolves experiment names to adapter classes
and instantiates fresh adapters on demand (no caching).
"""
from ex.utils.hpo.adapters.base import ExperimentAdapter
from ex.utils.hpo.adapters.mnist import MnistAdapter
from ex.utils.hpo.adapters.mnist_uncond import MnistUncondAdapter
from ex.utils.hpo.adapters.dbpedia import DbpediaAdapter
from ex.utils.hpo.adapters.pendulum import PendulumAdapter
from ex.utils.hpo.adapters.model_selection import ModelSelectionAdapter
from ex.utils.hpo.adapters.eig import EIGAdapter
from ex.utils.hpo.adapters.elbo import ELBOAdapter
from ex.utils.hpo.adapters.occupancy import OccupancyAdapter
from ex.utils.hpo.adapters.dre_sample_complexity import DreSampleComplexityAdapter
from ex.utils.hpo.adapters.dokls import DoklsAdapter




_ADAPTERS: dict[str, type[ExperimentAdapter]] = {
    # canonical (new) keys
    "mnist": MnistAdapter,
    "mnist_uncond": MnistUncondAdapter,
    "dbpedia": DbpediaAdapter,
    "pendulum": PendulumAdapter,
    "occupancy": OccupancyAdapter,
    # unrelated, no rename
    "model_selection": ModelSelectionAdapter,
    "eig": EIGAdapter,
    "elbo": ELBOAdapter,
    "dre_sample_complexity": DreSampleComplexityAdapter,
}


def get_adapter(name: str) -> ExperimentAdapter:
    """
    resolve adapter by name. instantiates a fresh adapter (no caching).

    args:
        name: experiment name (str key in registry, or model_selection_<tag>).

    returns:
        fresh instance of adapter class.

    raises:
        KeyError: if name not in registry and not a model_selection_* variant.
                  message lists known adapters and valid model_selection tags.

    behavior:
        1. if name in _ADAPTERS, instantiate and return.
        2. elif name.startswith("model_selection_"), resolve tag via
           variants.tag_from_experiment(name); get config_path via
           variants.config_path(tag); instantiate ModelSelectionAdapter with
           that config_path.
        3. else raise KeyError with helpful message listing known adapters.
    """
    if name in _ADAPTERS:
        return _ADAPTERS[name]()

    if name.startswith("model_selection_"):
        # lazy import to avoid circular dependency at module load time
        from ex.synth.model_selection import variants
        try:
            tag = variants.tag_from_experiment(name)
            config_path = variants.config_path(tag)
            return ModelSelectionAdapter(config_path=config_path)
        except KeyError:
            # tag_from_experiment raised KeyError for unknown tag
            valid_tags = ", ".join(sorted(variants.VARIANTS.keys()))
            raise KeyError(
                f"unknown model_selection variant: {name!r}; "
                f"valid tags: {valid_tags}"
            ) from None

    if name.startswith("dokls_"):
        # lazy import to avoid circular dependency at module load time.
        # experiment is dokls_{route}_{tag}; adapter is keyed by (pstar_idx, N, route).
        from ex.ablations.dokls import variants as dokls_variants
        try:
            route, tag = dokls_variants.tag_from_experiment(name)
            pstar_idx, N, nstar = dokls_variants.VARIANTS[tag]
            return DoklsAdapter(pstar_idx, N, nstar=nstar, route=route)
        except KeyError:
            valid_tags = ", ".join(sorted(dokls_variants.VARIANTS.keys()))
            raise KeyError(
                f"unknown dokls variant: {name!r}; valid tags: {valid_tags}; "
                f"routes: {dokls_variants.ROUTES}"
            ) from None

    # neither registry hit nor model_selection_* / dokls_* pattern
    known = sorted(_ADAPTERS)
    raise KeyError(f"unknown experiment: {name!r}; known: {known}")


def list_adapters() -> list[str]:
    """
    sorted list of registered adapter names. used by launcher --experiments all.

    returns:
        list[str]: sorted names of all registered adapters.

    use case:
        launcher's --experiments all flag; UI enumeration.
    """
    return sorted(_ADAPTERS.keys())
