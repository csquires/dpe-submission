"""Train and evaluate a regression method on a synthetic fixture.

Single entry point: train_then_eval() handles method dispatch, hyperparameter
merging, grid construction, training, and evaluation.
"""
import time
from dataclasses import dataclass

import torch

from tests.methods.reg._dispatch import get_method
from tests.methods.reg.harness import seed_everything, default_hp


@dataclass(frozen=True)
class EvalResult:
    """Result of training and evaluating a regression method.

    Attributes:
        mae: mean absolute error on eval grid. float('inf') if predictions contain NaN/Inf.
        predicted_ldr: detached, cpu, float32 tensor of shape [N] with model predictions.
        true_ldr: detached, cpu, float32 tensor of shape [N] with fixture ground truth.
        runtime_s: wall time from fit start to predict end, measured via time.perf_counter().
        meta: dict with keys method (str), D (int), fixture_meta (dict), n_epochs (int),
              hp_override (dict or empty dict).
    """
    mae: float
    predicted_ldr: torch.Tensor
    true_ldr: torch.Tensor
    runtime_s: float
    meta: dict


def train_then_eval(
    method: str,
    fixture,
    *,
    n_epochs: int = 800,
    hp_override: dict | None = None,
    eval_xs: torch.Tensor | None = None,
    seed: int = 0,
) -> EvalResult:
    """Train a regression method and evaluate on grid.

    Args:
        method: method name (e.g., "CTSM", "TriangularCTSM_V1").
        fixture: SyntheticFixture with samples_p0, samples_p1, samples_pstar (optional),
                 true_ldr callable, meta dict.
        n_epochs: number of training epochs (overrides hp["n_epochs"]).
        hp_override: dict of hyperparameters to merge into defaults.
        eval_xs: optional eval grid of shape [N, D], float32, cpu. if None, build
                 deterministic uniform grid over [-3, 3]^D capped at ~512 points.
        seed: random seed for reproducibility.

    Returns:
        EvalResult with mae, predicted_ldr, true_ldr, runtime_s, meta.

    Raises:
        ValueError: if method unknown, or if requires_pstar but fixture.samples_pstar is None.
    """
    # seed everything for reproducibility
    seed_everything(seed)

    # dispatch method
    dispatch = get_method(method)

    # extract dimension
    D = fixture.samples_p0.shape[1]

    # load and merge hyperparameters
    hp = default_hp(method)
    if hp_override is not None:
        hp.update(hp_override)
    hp["n_epochs"] = n_epochs

    # build model
    nw = hp.pop("num_waypoints", 3)
    est = dispatch["builder"](input_dim=D, device="cpu", num_waypoints=nw, **hp)

    # eval grid: deterministic uniform or provided
    if eval_xs is None:
        # build deterministic uniform grid over [-3, 3]^D
        k = max(2, int(512 ** (1.0 / D)))
        linspaces = [torch.linspace(-3, 3, k) for _ in range(D)]
        grids = torch.meshgrid(*linspaces, indexing='ij')
        eval_xs = torch.stack(grids, dim=-1)  # [k, k, ..., D]
        eval_xs = eval_xs.reshape(-1, D)  # [k^D, D]
    else:
        eval_xs = eval_xs.float().cpu()

    # fit: build args tuple
    fit_args = (fixture.samples_p0, fixture.samples_p1)
    if dispatch["requires_pstar"]:
        if fixture.samples_pstar is None:
            raise ValueError(
                f"method {method} requires_pstar but fixture.samples_pstar is None"
            )
        fit_args += (fixture.samples_pstar,)

    # time fit and predict
    t0 = time.perf_counter()
    est.fit(*fit_args)

    # predict
    pred = est.predict_ldr(eval_xs)
    pred = pred.detach().cpu().float()  # [N] batch
    true = fixture.true_ldr(eval_xs).detach().cpu().float()  # [N] batch
    t1 = time.perf_counter()

    # compute mae
    if torch.isfinite(pred).all():
        mae = (pred - true).abs().mean().item()
    else:
        mae = float('inf')

    # build meta
    meta = {
        "method": method,
        "D": D,
        "fixture_meta": fixture.meta,
        "n_epochs": n_epochs,
        "hp_override": hp_override or {},
    }

    return EvalResult(
        mae=mae,
        predicted_ldr=pred,
        true_ldr=true,
        runtime_s=t1 - t0,
        meta=meta
    )


def null_baseline_mae(
    fixture,
    eval_xs: torch.Tensor,
) -> float:
    """MAE of predicting zero everywhere.

    Args:
        fixture: SyntheticFixture with true_ldr callable.
        eval_xs: eval grid of shape [N, D].

    Returns:
        float: mean absolute value of fixture.true_ldr(eval_xs).
    """
    return fixture.true_ldr(eval_xs).abs().mean().item()
