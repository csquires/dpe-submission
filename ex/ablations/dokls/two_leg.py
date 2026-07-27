"""anchored and direct log-density-ratio estimators for dokls ablation."""
from typing import Callable

import torch

from src.methods.common.base import ELDR


def _at_rung(step: int, interval) -> bool:
    """report gate. hyperband passes the rung set (a frozenset of global steps);
    the non-pruning fallback passes an int cadence. handle both.
    """
    if isinstance(interval, (set, frozenset)):
        return step in interval
    return step % interval == 0


class TwoLegELDR(ELDR):
    """anchored-leg density-ratio estimator via two independent critics.

    trains two legs (leg0, leg1) on the SAME pstar samples paired with different
    comparison groups (p0, p1). the difference of their predictions recovers the ELDR.

    pseudocode:
      leg0 = leg_builder(...); leg0.init_fit(pstar, p0)  # log(p*/p0)
      leg1 = leg_builder(...); leg1.init_fit(pstar, p1)  # log(p*/p1)
      for g in range(n_steps):                       # 6400 iters
          (leg0 if g % 2 == 0 else leg1).train_step()  # -> 3200 steps / leg
          if _at_rung(g+1, step_cb_interval) and eval_fn:
              step_cb(g+1, eval_fn())
      leg0.finalize()
      leg1.finalize()
      predict_ldr(xs) = leg1.predict_ldr(xs) - leg0.predict_ldr(xs)
    """

    def __init__(
        self,
        leg_builder: Callable,
        input_dim: int,
        device: torch.device,
        n_steps: int = 6400,
        **hp
    ) -> None:
        """bind leg factory and hyperparameters.

        args:
            leg_builder: callable(input_dim, device, **hp) -> DRE with init_fit,
                train_step, predict_ldr, finalize.
            input_dim: feature dimension.
            device: torch.device.
            n_steps: total gradient steps; per-leg budget = n_steps // 2.
            **hp: hyperparameters passed to leg_builder.
        """
        super().__init__(input_dim)
        self.leg_builder = leg_builder
        self.device = device
        self.n_steps = n_steps
        self.hp = hp

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """train both legs via interleaved gradient steps.

        args:
            samples_p0: [N0, D] samples from p0.
            samples_p1: [N1, D] samples from p1.
            samples_pstar: [N*, D] reference samples (anchor for both legs).
            step_cb: optuna pruning hook fired every step_cb_interval outer iterations
                (1-indexed). None disables instrumentation.
            eval_data: dict with key "true_ldrs" -> [N*,] ground-truth log p0/p1 values.
                required if step_cb is not None.
            step_cb_interval: interval for firing step_cb.

        raises:
            ValueError: dimension mismatch, or step_cb not None but eval_data missing.
        """
        # dimension validation
        if (
            samples_p0.shape[1] != samples_p1.shape[1]
            or samples_p0.shape[1] != samples_pstar.shape[1]
        ):
            raise ValueError(
                f"dimension mismatch: p0 {samples_p0.shape[1]} vs p1 {samples_p1.shape[1]} "
                f"vs pstar {samples_pstar.shape[1]}"
            )

        # eval_data validation
        if step_cb is not None and eval_data is None:
            raise ValueError("step_cb is not None but eval_data is None")
        if step_cb is not None and "true_ldrs" not in eval_data:
            raise ValueError("step_cb is not None but eval_data missing 'true_ldrs'")

        # build and initialize legs
        self.leg0 = self.leg_builder(self.input_dim, self.device, **self.hp)
        self.leg1 = self.leg_builder(self.input_dim, self.device, **self.hp)

        self.leg0.init_fit(samples_pstar, samples_p0)  # log(p*/p0)
        self.leg1.init_fit(samples_pstar, samples_p1)  # log(p*/p1)

        # define eval function
        eval_fn = None
        if step_cb is not None and eval_data is not None:

            def eval_fn() -> torch.Tensor:
                with torch.no_grad():
                    preds = self.predict_ldr(samples_pstar)  # [N*,]
                    true_ldrs = eval_data["true_ldrs"]  # [N*,]
                    # predict_ldr returns cpu (reg methods .detach().cpu()); true_ldrs
                    # is on-device. match the final-metric convention (.cpu() both).
                    mae = torch.mean(torch.abs(preds.cpu() - true_ldrs.cpu()))
                    return mae

        # interleaved training loop: alternate ONE leg per iteration (vfm-style).
        # n_steps=6400 iters => 3200 steps/leg. report the outer iteration count
        # g+1 (the _make_report / _make_report_pair convention); at rung R each
        # leg has done ~R/2 steps, compute-matched to the direct arm's R.
        for g in range(self.n_steps):
            (self.leg0 if g % 2 == 0 else self.leg1).train_step()

            if step_cb is not None and eval_fn is not None and _at_rung(g + 1, step_cb_interval):
                step_cb(g + 1, float(eval_fn().item()))

        # finalize
        self.leg0.finalize()
        self.leg1.finalize()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """point-wise log-density-ratio: log(p0(xs)/p1(xs)).

        returns: [M,]; computed as leg1.predict_ldr(xs) - leg0.predict_ldr(xs).
        """
        pred0 = self.leg0.predict_ldr(xs)  # log(p*/p0)
        pred1 = self.leg1.predict_ldr(xs)  # log(p*/p1)
        return pred1 - pred0  # log(p0/p1)


class DirectELDR(ELDR):
    """single-leg density-ratio estimator via direct critic on (p0, p1) pairs.

    trains one critic directly on (p0, p1) pairs, bypassing p* anchoring. pstar is
    provided (to match ELDR contract) but NOT used during training; only eval_data
    participates in eval_fn.

    pseudocode:
      leg = leg_builder(...)
      leg.init_fit(p0, p1)  # log(p0/p1)
      for step in range(n_steps):
          leg.train_step()
          if (step+1) % step_cb_interval == 0 and eval_fn:
              step_cb(step+1, eval_fn())
      leg.finalize()
      predict_ldr(xs) = leg.predict_ldr(xs)
    """

    def __init__(
        self,
        leg_builder: Callable,
        input_dim: int,
        device: torch.device,
        n_steps: int = 6400,
        **hp
    ) -> None:
        """bind leg factory and hyperparameters (signature identical to TwoLegELDR).

        args:
            leg_builder: callable(input_dim, device, **hp) -> DRE with init_fit,
                train_step, predict_ldr, finalize.
            input_dim: feature dimension.
            device: torch.device.
            n_steps: total gradient steps (6400).
            **hp: hyperparameters passed to leg_builder.
        """
        super().__init__(input_dim)
        self.leg_builder = leg_builder
        self.device = device
        self.n_steps = n_steps
        self.hp = hp

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """train single leg on (p0, p1) pairs.

        args:
            samples_p0: [N0, D] numerator group.
            samples_p1: [N1, D] denominator group.
            samples_pstar: [N*, D] reference samples; ignored during fit (included
                in contract for polymorphism with TwoLegELDR).
            step_cb: optuna pruning hook fired every step_cb_interval iterations
                (1-indexed). None disables instrumentation.
            eval_data: dict with key "true_ldrs" -> [N*,] ground-truth log p0/p1 values.
                required if step_cb is not None.
            step_cb_interval: interval for firing step_cb.

        raises:
            ValueError: dimension mismatch, or step_cb not None but eval_data missing.
        """
        # dimension validation
        if samples_p0.shape[1] != samples_p1.shape[1]:
            raise ValueError(
                f"dimension mismatch: p0 {samples_p0.shape[1]} vs p1 {samples_p1.shape[1]}"
            )

        # eval_data validation
        if step_cb is not None and eval_data is None:
            raise ValueError("step_cb is not None but eval_data is None")
        if step_cb is not None and "true_ldrs" not in eval_data:
            raise ValueError("step_cb is not None but eval_data missing 'true_ldrs'")

        # build and initialize leg
        self.leg = self.leg_builder(self.input_dim, self.device, **self.hp)
        self.leg.init_fit(samples_p0, samples_p1)  # log(p0/p1)

        # define eval function
        eval_fn = None
        if step_cb is not None and eval_data is not None:

            def eval_fn() -> torch.Tensor:
                with torch.no_grad():
                    preds = self.leg.predict_ldr(samples_pstar)  # [N*,]
                    true_ldrs = eval_data["true_ldrs"]  # [N*,]
                    # predict_ldr returns cpu (reg methods .detach().cpu()); true_ldrs
                    # is on-device. match the final-metric convention (.cpu() both).
                    mae = torch.mean(torch.abs(preds.cpu() - true_ldrs.cpu()))
                    return mae

        # single-leg training loop
        for step in range(self.n_steps):
            self.leg.train_step()

            if step_cb is not None and eval_fn is not None and _at_rung(step + 1, step_cb_interval):
                step_cb(step + 1, float(eval_fn().item()))

        # finalize
        self.leg.finalize()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """point-wise log-density-ratio: log(p0(xs)/p1(xs)).

        returns: [M,]; delegates to leg.predict_ldr(xs).
        """
        return self.leg.predict_ldr(xs)
