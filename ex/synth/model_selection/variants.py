"""single source of truth for model_selection variant registry and paths.

ensures full isolation: distinct on-disk paths (configs/, winners/), hpo modules,
and validation. registry is explicit (dict only), not filesystem-scanned.
"""

import os
from pathlib import Path

import h5py

from src.utils.io import _load_config


VARIANTS: dict[str, tuple[int, int]] = {
    "tr8192_te2048": (8192, 2048),
    "tr8192_te4096": (8192, 4096),
    "tr8192_te8192": (8192, 8192),
    "tr8192_te16384": (8192, 16384),
}

DEFAULT_TAG: str = "tr8192_te2048"


def experiment_name(tag: str) -> str:
    """canonical experiment name for this tag.

    args:
        tag: variant tag (e.g., "tr8192_te2048")

    returns:
        f"model_selection_{tag}"
    """
    return f"model_selection_{tag}"


def tag_from_experiment(name: str) -> str:
    """extract tag from experiment name (inverse of experiment_name).

    args:
        name: experiment name (e.g., "model_selection_tr8192_te2048")

    returns:
        tag (e.g., "tr8192_te2048")

    raises:
        ValueError: if name doesn't start with "model_selection_"
        KeyError: if extracted tag not in VARIANTS
    """
    if not name.startswith("model_selection_"):
        raise ValueError(
            f"experiment name must start with 'model_selection_', got {name!r}"
        )
    tag = name[len("model_selection_") :]
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
        )
    return tag


def config_path(tag: str) -> Path:
    """absolute path to yaml config for this variant.

    args:
        tag: variant tag (e.g., "tr8192_te2048")

    returns:
        Path to configs/<tag>.yaml (absolute)

    raises:
        KeyError: if tag not in VARIANTS
    """
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
        )
    base = Path(__file__).resolve().parent
    return base / "configs" / f"{tag}.yaml"


def winners_path(tag: str) -> Path:
    """absolute path to yaml winners for this variant.

    args:
        tag: variant tag (e.g., "tr8192_te2048")

    returns:
        Path to winners/<tag>.yaml (absolute)

    raises:
        KeyError: if tag not in VARIANTS
    """
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
        )
    base = Path(__file__).resolve().parent
    return base / "winners" / f"{tag}.yaml"


def resolve(variant: str | None = None) -> tuple[str, dict]:
    """resolve variant via precedence: explicit arg > env > default. load config.

    args:
        variant: optional explicit variant tag. if provided, must be in VARIANTS.

    returns:
        (tag, config_dict) where config is loaded with env var expansion.

    raises:
        KeyError: if explicit or env variant not in VARIANTS
        FileNotFoundError: if config yaml doesn't exist
        RuntimeError: if required env roots unset
    """
    # determine tag via precedence: arg > env > default
    if variant is not None:
        tag = variant
        if tag not in VARIANTS:
            raise KeyError(
                f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
            )
    else:
        env_variant = os.environ.get("DPE_MS_VARIANT")
        if env_variant is not None:
            tag = env_variant
            if tag not in VARIANTS:
                raise KeyError(
                    f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
                )
        else:
            tag = DEFAULT_TAG

    print(f"Resolved model_selection variant: {tag}")

    # load config
    config = _load_config(str(config_path(tag)))
    return tag, config


def assert_dataset_matches(config: dict, h5_path: str | Path) -> None:
    """validate that dataset shapes match config sample counts.

    opens h5; checks samples_p0_arr.shape[1] == config['nsamples_train'],
    samples_pstar_arr.shape[2] == config['nsamples_test'].

    args:
        config: dict with keys 'nsamples_train' and 'nsamples_test'
        h5_path: path to h5 file

    raises:
        ValueError: if shapes don't match config (includes both variant and actual)
    """
    with h5py.File(h5_path, "r") as f:
        n_train_actual = f["samples_p0_arr"].shape[1]
        n_test_actual = f["samples_pstar_arr"].shape[2]

    if (
        config["nsamples_train"] != n_train_actual
        or config["nsamples_test"] != n_test_actual
    ):
        raise ValueError(
            f"dataset shape mismatch for variant config "
            f"(nsamples_train={config['nsamples_train']}, nsamples_test={config['nsamples_test']}): "
            f"h5 has samples_p0_arr.shape[1]={n_train_actual}, samples_pstar_arr.shape[2]={n_test_actual}"
        )


def hpo_config_modules(tag: str) -> list[str]:
    """list hpo studyconfig module paths for this tag.

    args:
        tag: variant tag (e.g., "tr8192_te2048")

    returns:
        list of two dotted module paths (cpu and gpu variants)

    raises:
        KeyError: if tag not in VARIANTS
    """
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
        )
    return [
        f"ex.utils.hpo.optuna.configs.model_selection_{tag}_cpu",
        f"ex.utils.hpo.optuna.configs.model_selection_{tag}_gpu",
    ]
