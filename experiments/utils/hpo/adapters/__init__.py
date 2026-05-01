"""
adapter factory: registry and factory functions.

minimal factory pattern. resolves experiment names to adapter classes
and instantiates fresh adapters on demand (no caching).
"""
from experiments.utils.hpo.adapters.base import ExperimentAdapter
from experiments.utils.hpo.adapters.mnist_cond_flow import MnistCondFlowAdapter
from experiments.utils.hpo.adapters.mnist_estimation import MnistEstimationAdapter
from experiments.utils.hpo.adapters.dbpedia_cond_flow import DbpediaCondFlowAdapter
from experiments.utils.hpo.adapters.pendulum_eldr_estimation import PendulumAdapter
from experiments.utils.hpo.adapters.model_selection import ModelSelectionAdapter
from experiments.utils.hpo.adapters.eig_estimation import EIGAdapter
from experiments.utils.hpo.adapters.elbo_estimation import ELBOAdapter
from experiments.utils.hpo.adapters.smodice_eldr_estimation import SmodiceAdapter

# from experiments.utils.hpo.adapters.pendulum_eldr_estimation import PendulumAdapter
# from experiments.utils.hpo.adapters.model_selection import ModelSelectionAdapter
# from experiments.utils.hpo.adapters.eig_estimation import EIGAdapter
# from experiments.utils.hpo.adapters.elbo_estimation import ELBOAdapter
# from experiments.utils.hpo.adapters.smodice_eldr_estimation import SmodiceAdapter


_ADAPTERS: dict[str, type[ExperimentAdapter]] = {
    "mnist_cond_flow": MnistCondFlowAdapter,
    "mnist_estimation": MnistEstimationAdapter,
    "dbpedia_cond_flow": DbpediaCondFlowAdapter,
    "pendulum_eldr_estimation": PendulumAdapter,
    "model_selection": ModelSelectionAdapter,
    "eig_estimation": EIGAdapter,
    "elbo_estimation": ELBOAdapter,
    "smodice_eldr_estimation": SmodiceAdapter,
    # "dbpedia_cond_flow": DbpediaCondFlowAdapter,
    # "pendulum_eldr_estimation": PendulumAdapter,
    # "model_selection": ModelSelectionAdapter,
    # "eig_estimation": EIGAdapter,
    # "elbo_estimation": ELBOAdapter,
    # "smodice_eldr_estimation": SmodiceAdapter,
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
