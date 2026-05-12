"""step2_runner adapter for mnist_eldr.

cell axis: flat int over (alpha_idx, pair_idx). bucket axis: alpha_idx_<n>.
"""
from experiments.utils.step2_runner.adapter_base import make_adapter_module
from experiments.utils.step2_runner.adapter_specs import MNIST_COND
from experiments.mnist.hpo_search_spaces import SEARCH_SPACES

_module = make_adapter_module(MNIST_COND, SEARCH_SPACES)

load_config = _module["load_config"]
list_cells = _module["list_cells"]
bucket_for_cell = _module["bucket_for_cell"]
fit_and_eval = _module["fit_and_eval"]
walltime_per_cell_seconds = _module["walltime_per_cell_seconds"]
resources_for_method = _module["resources_for_method"]
is_cpu_eligible = _module["is_cpu_eligible"]
method_label = _module["method_label"]
gather_dataset_name = _module["gather_dataset_name"]
gather_output_path = _module["gather_output_path"]
