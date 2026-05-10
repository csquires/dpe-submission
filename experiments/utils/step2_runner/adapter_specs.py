from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AdapterSpec:
    """Consolidated per-experiment step2 adapter configuration.

    Fields:
    - name: experiment label for logging/errors
    - walltimes: dict mapping 'fast'/'medium'/'slow'/'default' to seconds
    - mem_slow: slurm resource string for slow methods
    - mem_default: slurm resource string for default methods
    - default_output_dir: fallback output path (used as config.get(..., default_output_dir))
    - input_dim_fn: function from config dict to input dimension
    """
    name: str
    walltimes: dict[str, int]
    mem_slow: str
    mem_default: str
    default_output_dir: str
    input_dim_fn: Callable[[dict], int]


MNIST_UNCOND = AdapterSpec(
    name="mnist_uncond",
    walltimes={"fast": 60, "medium": 120, "slow": 240, "default": 120},
    mem_slow="--gpus=1 --cpus-per-task=4 --mem=24G",
    mem_default="--gpus=1 --cpus-per-task=2 --mem=16G",
    default_output_dir="experiments/mnist_uncond/raw_results",
    input_dim_fn=lambda c: c["latent_dim"],
)

MNIST_COND = AdapterSpec(
    name="mnist_eldr",
    walltimes={"fast": 30, "medium": 60, "slow": 120, "default": 60},
    mem_slow="--gpus=1 --cpus-per-task=4 --mem=16G",
    mem_default="--gpus=1 --cpus-per-task=2 --mem=16G",
    default_output_dir="experiments/mnist_eldr/raw_results",
    input_dim_fn=lambda c: c["latent_dim"],
)

DBPEDIA_COND = AdapterSpec(
    name="dbpedia_eldr",
    walltimes={"fast": 60, "medium": 120, "slow": 180, "default": 120},
    mem_slow="--gpus=1 --cpus-per-task=4 --mem=16G",
    mem_default="--gpus=1 --cpus-per-task=2 --mem=16G",
    default_output_dir="experiments/dbpedia_eldr/raw_results",
    input_dim_fn=lambda c: c.get("pca_dim", c["latent_dim"]),
)
