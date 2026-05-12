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
        name: experiment name (str key in registry).

    returns:
        fresh instance of adapter class.

    raises:
        KeyError: if name not in registry. message lists all known adapters.

    behavior:
        1. check if name in _ADAPTERS; raise KeyError with helpful message if not.
        2. retrieve adapter class.
        3. instantiate with no args.
        4. return fresh instance (no caching; each call creates new instance).
    """
    if name not in _ADAPTERS:
        known = sorted(_ADAPTERS)
        raise KeyError(f"unknown experiment: {name!r}; known: {known}")

    return _ADAPTERS[name]()


def list_adapters() -> list[str]:
    """
    sorted list of registered adapter names. used by launcher --experiments all.

    returns:
        list[str]: sorted names of all registered adapters.

    use case:
        launcher's --experiments all flag; UI enumeration.
    """
    return sorted(_ADAPTERS.keys())
