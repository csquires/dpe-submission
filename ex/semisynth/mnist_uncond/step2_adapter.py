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


def gather_grid_size(config: dict) -> int:
    """full-grid size for gather = n_alphas * num_pairs_per_alpha.

    step2 runs on the SPARSE disjoint step2_pool (a subset of the full
    alpha x pair grid; train/holdout cells removed within each alpha), so
    gather must fill the FULL grid at true cell_idx (= alpha*num_pairs + pair)
    and NaN-fill the held-out cells. otherwise range(len(list_cells)) reads
    the wrong cells and mis-aligns them to step3's i//num_pairs alpha rows.
    mirrors the elbo sparse-grid handling.
    """
    return len(config["alphas"]) * config["num_pairs_per_alpha"]
