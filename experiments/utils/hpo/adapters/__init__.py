"""
adapter factory: registry and factory functions.

minimal factory pattern. resolves experiment names to adapter classes
and instantiates fresh adapters on demand (no caching).
"""
from experiments.utils.hpo.adapters.base import ExperimentAdapter
from experiments.utils.hpo.adapters.mnist import MnistAdapter
from experiments.utils.hpo.adapters.mnist_uncond import MnistUncondAdapter
from experiments.utils.hpo.adapters.dbpedia import DbpediaAdapter
from experiments.utils.hpo.adapters.pendulum import PendulumAdapter
from experiments.utils.hpo.adapters.model_selection import ModelSelectionAdapter
from experiments.utils.hpo.adapters.eig_estimation import EIGAdapter
from experiments.utils.hpo.adapters.elbo_estimation import ELBOAdapter
from experiments.utils.hpo.adapters.occupancy import OccupancyAdapter
from experiments.utils.hpo.adapters.dre_sample_complexity import DreSampleComplexityAdapter




_ADAPTERS: dict[str, type[ExperimentAdapter]] = {
    # canonical (new) keys
    "mnist": MnistAdapter,
    "mnist_uncond": MnistUncondAdapter,
    "dbpedia": DbpediaAdapter,
    "pendulum": PendulumAdapter,
    "occupancy": OccupancyAdapter,
    # unrelated, no rename
    "model_selection": ModelSelectionAdapter,
    "eig_estimation": EIGAdapter,
    "elbo_estimation": ELBOAdapter,
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
