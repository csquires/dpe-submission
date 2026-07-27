"""single source of truth for dokls ablation variant registry and paths.

ensures full isolation: distinct on-disk paths (configs/, winners/), hpo modules,
and validation. registry is explicit (dict only), not filesystem-scanned.
"""

import os
from pathlib import Path

import h5py

from src.utils.io import _load_config


# tag -> (pstar_idx, N, Nstar). N = p0/p1 sample budget (density-ratio fit);
# Nstar = p* sample budget (leg anchoring + MC average). the original 8 variants
# are the diagonal Nstar == N; the *_ns<Nstar> tags decouple them to backfill the
# N=8192 column (matches model_selection's fixed-N-train sweep of Nstar).
VARIANTS: dict[str, tuple[int, int, int]] = {
    "q0_N1024": (0, 1024, 1024),
    "q0_N2048": (0, 2048, 2048),
    "q0_N4096": (0, 4096, 4096),
    "q0_N8192": (0, 8192, 8192),
    "q1_N1024": (1, 1024, 1024),
    "q1_N2048": (1, 2048, 2048),
    "q1_N4096": (1, 4096, 4096),
    "q1_N8192": (1, 8192, 8192),
    # decoupled: N=8192 p0/p1, smaller Nstar p* (comparable to MS te2048/te4096).
    "q0_N8192_ns2048": (0, 8192, 2048),
    "q0_N8192_ns4096": (0, 8192, 4096),
    "q1_N8192_ns2048": (1, 8192, 2048),
    "q1_N8192_ns4096": (1, 8192, 4096),
}

ROUTES: list[str] = ["two_leg", "direct"]

DEFAULT_TAG: str = "q0_N1024"
DEFAULT_ROUTE: str = "two_leg"


def experiment_name(tag: str, route: str) -> str:
    """canonical experiment name for this tag and route.

    args:
        tag: variant tag (e.g., "q0_N1024")
        route: routing scheme ("two_leg" or "direct")

    returns:
        f"dokls_{route}_{tag}"
    """
    return f"dokls_{route}_{tag}"


def tag_from_experiment(name: str) -> tuple[str, str]:
    """extract (route, tag) from experiment name (inverse of experiment_name).

    parses "dokls_{route}_{tag}" with literal prefix test (not naive split on "_").

    args:
        name: experiment name (e.g., "dokls_two_leg_q0_N1024")

    returns:
        (route, tag) (e.g., ("two_leg", "q0_N1024"))

    raises:
        KeyError: if name doesn't start with "dokls_", if no route prefix matches,
                  or if extracted tag not in VARIANTS
    """
    if not name.startswith("dokls_"):
        raise KeyError(
            f"experiment name must start with 'dokls_', got {name!r}"
        )
    rest = name[len("dokls_") :]

    # literal prefix test: check each route as full prefix with underscore.
    # critical: "two_leg" itself contains underscore, so cannot use naive split.
    route_found = None
    for route in ROUTES:
        if rest.startswith(f"{route}_"):
            route_found = route
            break

    if route_found is None:
        raise KeyError(
            f"no known route prefix matches {name!r}; valid routes: {ROUTES}"
        )

    tag = rest[len(route_found) + 1 :]
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r} in {name!r}; valid tags: {sorted(VARIANTS.keys())}"
        )

    return route_found, tag


def config_path(tag: str, route: str) -> Path:
    """absolute path to yaml config for this variant and route.

    args:
        tag: variant tag (e.g., "q0_N1024")
        route: routing scheme ("two_leg" or "direct")

    returns:
        Path to ex/ablations/dokls/configs/dokls_<route>_<tag>.yaml (absolute)

    raises:
        KeyError: if tag not in VARIANTS or route not in ROUTES
    """
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
        )
    if route not in ROUTES:
        raise KeyError(
            f"unknown route {route!r}; valid routes: {ROUTES}"
        )
    base = Path(__file__).resolve().parent
    return base / "configs" / f"dokls_{route}_{tag}.yaml"


def winners_path(tag: str, route: str) -> Path:
    """absolute path to yaml winners for this variant and route.

    args:
        tag: variant tag (e.g., "q0_N1024")
        route: routing scheme ("two_leg" or "direct")

    returns:
        Path to ex/ablations/dokls/winners/dokls_<route>_<tag>.yaml (absolute)

    raises:
        KeyError: if tag not in VARIANTS or route not in ROUTES
    """
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
        )
    if route not in ROUTES:
        raise KeyError(
            f"unknown route {route!r}; valid routes: {ROUTES}"
        )
    base = Path(__file__).resolve().parent
    return base / "winners" / f"dokls_{route}_{tag}.yaml"


def hpo_config_modules(tag: str, route: str) -> list[str]:
    """list hpo studyconfig module paths for this variant and route.

    args:
        tag: variant tag (e.g., "q0_N1024")
        route: routing scheme ("two_leg" or "direct")

    returns:
        list of two dotted module paths (cpu and gpu variants)

    raises:
        KeyError: if tag not in VARIANTS or route not in ROUTES
    """
    if tag not in VARIANTS:
        raise KeyError(
            f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
        )
    if route not in ROUTES:
        raise KeyError(
            f"unknown route {route!r}; valid routes: {ROUTES}"
        )
    return [
        f"ex.utils.hpo.optuna.configs.dokls_{route}_{tag}_cpu",
        f"ex.utils.hpo.optuna.configs.dokls_{route}_{tag}_gpu",
    ]


def resolve(tag: str | None = None, route: str | None = None) -> tuple[str, str, dict]:
    """resolve variant+route via precedence: explicit arg > env > default. load config.

    args:
        tag: optional explicit variant tag. if provided, must be in VARIANTS.
        route: optional explicit route. if provided, must be in ROUTES.

    returns:
        (tag, route, config_dict) where config is loaded with env var expansion

    raises:
        KeyError: if explicit or env tag/route not in VARIANTS/ROUTES
        FileNotFoundError: if config yaml doesn't exist
        RuntimeError: if required env roots unset during _load_config
    """
    # determine tag via precedence: arg > env > default
    if tag is not None:
        if tag not in VARIANTS:
            raise KeyError(
                f"unknown variant {tag!r}; valid tags: {sorted(VARIANTS.keys())}"
            )
    else:
        env_tag = os.environ.get("DPE_DOKLS_TAG")
        if env_tag is not None:
            if env_tag not in VARIANTS:
                raise KeyError(
                    f"unknown variant {env_tag!r}; valid tags: {sorted(VARIANTS.keys())}"
                )
            tag = env_tag
        else:
            tag = DEFAULT_TAG

    # determine route via precedence: arg > env > default
    if route is not None:
        if route not in ROUTES:
            raise KeyError(
                f"unknown route {route!r}; valid routes: {ROUTES}"
            )
    else:
        env_route = os.environ.get("DPE_DOKLS_ROUTE")
        if env_route is not None:
            if env_route not in ROUTES:
                raise KeyError(
                    f"unknown route {env_route!r}; valid routes: {ROUTES}"
                )
            route = env_route
        else:
            route = DEFAULT_ROUTE

    print(f"Resolved dokls variant: {tag} route: {route}")

    # load config
    config = _load_config(str(config_path(tag, route)))
    return tag, route, config
