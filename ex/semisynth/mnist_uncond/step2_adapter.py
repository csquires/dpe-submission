"""step2_runner adapter for mnist_uncond.

cell axis: flat int over (alpha_idx, pair_idx). 4 alphas x 40 pairs = 160 cells.
bucket axis: f"alpha_idx_{alpha_idx}". input_dim: config['latent_dim'].
"""
from ex.utils.step2_runner.adapter_base import make_adapter_module
from ex.utils.step2_runner.adapter_specs import MNIST_UNCOND
from ex.utils.hpo.method_specs import METHOD_SPECS

_module = make_adapter_module(MNIST_UNCOND, METHOD_SPECS)

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
