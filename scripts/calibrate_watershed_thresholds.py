"""calibrate watershed thresholds by running train_then_eval per (method, fixture, D)."""
import argparse
import logging
import os
import subprocess
import sys
from datetime import date

import yaml

from tests.methods.reg._dispatch import FLOW_METHODS, is_triangular
from tests.methods.reg.fixtures import (
    gaussian_gaussian, heteroscedastic, triangular_gaussian
)
from tests.methods.reg.train_eval import train_then_eval


logger = logging.getLogger(__name__)


def parse_args():
    """parse CLI arguments for threshold calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate watershed thresholds by running train_then_eval per (method, fixture, D)."
    )

    parser.add_argument(
        "--out",
        type=str,
        default="tests/methods/reg/thresholds.yaml",
        help="Output YAML file path (default: tests/methods/reg/thresholds.yaml)."
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=800,
        help="Number of training epochs per run (default: 800)."
    )
    parser.add_argument(
        "--fmdre-multiplier",
        type=float,
        default=5.0,
        help="Multiplier on FMDRE MAE floor for non-FMDRE methods (default: 5.0)."
    )
    parser.add_argument(
        "--regression-margin",
        type=float,
        default=1.1,
        help="Multiplier on per-method MAE for tighter threshold (default: 1.1)."
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Subset of methods to calibrate (default: all FLOW_METHODS). Space-separated."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print threshold table without writing YAML."
    )

    return parser.parse_args()


def build_sweep(methods_filter: list[str] | None = None) -> list[tuple[str, str, int]]:
    """construct (method, fixture_key, D) tuples respecting triangular compatibility.

    Args:
        methods_filter: If provided, only include methods in this list. If None, use all FLOW_METHODS.

    Returns:
        List of (method_name, fixture_key, D) tuples:
        - gaussian_gaussian: all methods, D in {2, 10}
        - heteroscedastic: all methods, D = 2
        - triangular_gaussian: only triangular methods (is_triangular(name) == True), D = 2
    """
    methods = methods_filter if methods_filter is not None else list(FLOW_METHODS)
    methods = [m for m in methods if m in FLOW_METHODS]  # validate

    sweep = []

    # gaussian_gaussian: D in {2, 10}
    for D in [2, 10]:
        for method in methods:
            sweep.append((method, "gaussian_gaussian", D))

    # heteroscedastic: D = 2
    for method in methods:
        sweep.append((method, "heteroscedastic", 2))

    # triangular_gaussian: D = 2, only triangular methods
    for method in methods:
        if is_triangular(method):
            sweep.append((method, "triangular_gaussian", 2))

    return sweep


def run_calibration(
    sweep: list[tuple[str, str, int]],
    n_epochs: int,
) -> dict[tuple[str, str, int], float]:
    """run train_then_eval for each (method, fixture_key, D) tuple.

    Args:
        sweep: list of (method_name, fixture_key, D) tuples
        n_epochs: epochs to pass to train_then_eval

    Returns:
        Dict mapping (method, fixture_key, D) -> mae (float, possibly inf).
    """
    mae_map = {}

    for method, fixture_key, D in sweep:
        print(f"[calibrate] {method:25s} {fixture_key:20s} D={D} ... ", end="", flush=True)

        try:
            # instantiate fixture
            if fixture_key == "gaussian_gaussian":
                fixture = gaussian_gaussian(D=D)
            elif fixture_key == "heteroscedastic":
                fixture = heteroscedastic(D=D)
            elif fixture_key == "triangular_gaussian":
                fixture = triangular_gaussian(D=D)
            else:
                raise ValueError(f"unknown fixture_key: {fixture_key}")

            # run train_then_eval
            result = train_then_eval(method, fixture, n_epochs=n_epochs)
            mae = result.mae

            mae_map[(method, fixture_key, D)] = mae
            print(f"mae={mae:.6f}")

        except Exception as e:
            print(f"FAILED: {e}")
            mae_map[(method, fixture_key, D)] = float('inf')

    return mae_map


def compute_thresholds(
    mae_map: dict[tuple[str, str, int], float],
    fmdre_multiplier: float,
    regression_margin: float,
) -> dict[str, dict[int, dict[str, float]]]:
    """derive threshold dict from MAE map using FMDRE anchor.

    Args:
        mae_map: (method, fixture_key, D) -> mae
        fmdre_multiplier: multiplier on FMDRE MAE floor (typically 5.0)
        regression_margin: multiplier on per-method MAE (typically 1.1)

    Returns:
        Dict structure: {fixture_key: {D: {method: threshold}}}
        Logs warnings for inf MAEs and fixture-group skips.
    """
    thresholds = {}

    # group by fixture_key to extract FMDRE MAE
    fixtures = {}
    for (method, fixture_key, D), mae in mae_map.items():
        if fixture_key not in fixtures:
            fixtures[fixture_key] = {}
        if D not in fixtures[fixture_key]:
            fixtures[fixture_key][D] = {}
        fixtures[fixture_key][D][method] = mae

    # for each fixture group: extract FMDRE, compute thresholds
    for fixture_key in sorted(fixtures.keys()):
        thresholds[fixture_key] = {}

        for D in sorted(fixtures[fixture_key].keys()):
            methods_mae = fixtures[fixture_key][D]
            fmdre_mae = methods_mae.get("FMDRE", float('inf'))

            if fmdre_mae == float('inf'):
                logger.warning(
                    f"FMDRE failed for fixture_key={fixture_key}, D={D}; "
                    f"skipping threshold derivation for this group."
                )
                continue

            thresholds[fixture_key][D] = {}

            for method in sorted(methods_mae.keys()):
                method_mae = methods_mae[method]

                if method == "FMDRE":
                    # FMDRE threshold = method_mae * 1.0 (+ optional tiny slack)
                    threshold = method_mae + 1e-6
                else:
                    if method_mae == float('inf'):
                        # broken method
                        logger.warning(
                            f"{method} failed for fixture_key={fixture_key}, D={D}; "
                            f"setting threshold=999.0"
                        )
                        threshold = 999.0
                    else:
                        # threshold = max(fmdre_floor, per_method_margin)
                        fmdre_floor = fmdre_multiplier * fmdre_mae
                        per_method_ceiling = regression_margin * method_mae
                        threshold = max(fmdre_floor, per_method_ceiling)

                thresholds[fixture_key][D][method] = threshold

    return thresholds


def merge_thresholds_into_yaml(
    new_thresholds: dict[str, dict[int, dict[str, float]]],
    yaml_path: str,
    mae_map: dict[tuple[str, str, int], float],
    fmdre_multiplier: float,
    regression_margin: float,
) -> dict:
    """load existing YAML (if present), merge new thresholds, add provenance.

    Args:
        new_thresholds: computed thresholds dict
        yaml_path: path to thresholds.yaml
        mae_map: original (method, fixture_key, D) -> mae map
        fmdre_multiplier: multiplier used (for provenance derivation string)
        regression_margin: margin multiplier used

    Returns:
        Complete dict ready to serialize: {provenance, thresholds}.
        If yaml_path doesn't exist, initializes fresh.
    """
    # load existing yaml if present
    existing = {}
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                existing = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse {yaml_path}: {e}; starting fresh.")
            existing = {}

    # merge thresholds (overwrite only calibrated entries; preserve unrelated)
    if "thresholds" not in existing:
        existing["thresholds"] = {}

    for fixture_key, d_dict in new_thresholds.items():
        if fixture_key not in existing["thresholds"]:
            existing["thresholds"][fixture_key] = {}
        for D, method_dict in d_dict.items():
            existing["thresholds"][fixture_key][D] = method_dict

    # extract FMDRE MAE per fixture for provenance
    fmdre_mae_per_fixture = {}
    for fixture_key in sorted(new_thresholds.keys()):
        for D in sorted(new_thresholds[fixture_key].keys()):
            fmdre_mae = mae_map.get(("FMDRE", fixture_key, D), float('inf'))
            key = f"{fixture_key}_D{D}"
            fmdre_mae_per_fixture[key] = fmdre_mae

    # get git commit hash
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    # update provenance
    existing["provenance"] = {
        "calibrated_date": date.today().isoformat(),
        "commit": commit,
        "fmdre_mae_per_fixture": fmdre_mae_per_fixture,
        "derivation": f"max({fmdre_multiplier}*FMDRE-MAE, {regression_margin}*method-MAE); FMDRE={1.0}*FMDRE-MAE+1e-6",
    }

    return existing


def print_table(
    thresholds: dict[str, dict[int, dict[str, float]]],
    mae_map: dict[tuple[str, str, int], float],
) -> None:
    """print ASCII table of (method, fixture_key, D, mae, threshold).

    Useful for --dry-run and debugging.
    """
    print("\n" + "=" * 100)
    print(f"{'Method':25} {'Fixture':20} {'D':5} {'MAE':12} {'Threshold':12}")
    print("-" * 100)

    for fixture_key in sorted(thresholds.keys()):
        for D in sorted(thresholds[fixture_key].keys()):
            for method in sorted(thresholds[fixture_key][D].keys()):
                mae = mae_map.get((method, fixture_key, D), float('inf'))
                threshold = thresholds[fixture_key][D][method]
                print(
                    f"{method:25} {fixture_key:20} {D:5} {mae:12.6f} {threshold:12.6f}"
                )

    print("=" * 100 + "\n")


def main():
    """main entry point for threshold calibration runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    args = parse_args()

    # validate fac env
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed. Activate 'fac' conda env: conda activate fac")
        sys.exit(1)

    # build sweep
    methods_filter = args.methods if args.methods else None
    try:
        sweep = build_sweep(methods_filter=methods_filter)
    except ValueError as e:
        logger.error(f"Sweep construction failed: {e}")
        sys.exit(1)

    logger.info(f"Sweep: {len(sweep)} (method, fixture, D) combinations")

    # run calibration
    logger.info("Running train_then_eval ...")
    mae_map = run_calibration(sweep, n_epochs=args.n_epochs)

    # compute thresholds
    logger.info("Deriving thresholds ...")
    thresholds = compute_thresholds(
        mae_map,
        fmdre_multiplier=args.fmdre_multiplier,
        regression_margin=args.regression_margin,
    )

    # print table
    print_table(thresholds, mae_map)

    # merge and write
    if args.dry_run:
        logger.info("--dry-run: skipping YAML write.")
    else:
        logger.info(f"Merging and writing to {args.out} ...")
        merged = merge_thresholds_into_yaml(
            thresholds, args.out, mae_map,
            fmdre_multiplier=args.fmdre_multiplier,
            regression_margin=args.regression_margin,
        )

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, 'w') as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

        n_thresholds = sum(
            len(thresholds[fixture_key][D])
            for fixture_key in thresholds
            for D in thresholds[fixture_key]
        )
        logger.info(f"Wrote {n_thresholds} thresholds to {args.out}")


if __name__ == "__main__":
    main()
