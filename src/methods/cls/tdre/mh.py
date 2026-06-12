from typing import Callable

import torch

from ...common.base import DRE
from src.models.binary_classification.multi_head_binary_classifier import (
    MultiHeadBinaryClassifier,
)
from src.waypoints.waypoints1d import WaypointBuilder1D, DefaultWaypointBuilder1D


class MultiHeadTDRE(DRE):
    """
    TDRE variant using a single multi-head binary classifier in place of a list
    of classifiers. Head i discriminates waypoint i (numerator) from waypoint
    i+1 (denominator); all heads train simultaneously over a shared backbone.
    """

    def __init__(
        self,
        classifier: MultiHeadBinaryClassifier,
        waypoint_builder: WaypointBuilder1D = DefaultWaypointBuilder1D(),
        num_waypoints: int = 10,
        device: str = "cuda",
        early_stop_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.num_waypoints = num_waypoints
        self.classifier = classifier.to(self.device)
        self.waypoint_builder = waypoint_builder
        self.early_stop_cfg = early_stop_cfg

        if self.classifier.num_heads != num_waypoints - 1:
            raise ValueError(
                "MultiHeadBinaryClassifier must have num_heads == num_waypoints - 1"
            )

    def fit(
        self,
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor,  # [b1, dim]
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """
        Fit multi-head classifier on waypoint pairs.

        args:
            samples_p0: samples from p0, shape [n0, dim]
            samples_p1: samples from p1, shape [n1, dim]
            step_cb: optional callback(step, score) invoked at intervals during training
            eval_data: optional dict with keys "pstar" and "true_ldrs" for evaluation
            step_cb_interval: number of steps between callback invocations (default: 50)
        """
        # initialize metadata container for early stopping info
        meta_out: dict = {}

        # build waypoints: [num_waypoints, batch_size, dim]
        waypoint_samples = self.waypoint_builder.build_waypoints(
            samples_p0, samples_p1, self.num_waypoints
        )  # [w, b, dim]
        b = waypoint_samples.shape[1]  # batch size

        # prepare training data for each head
        xs_per_head = []
        ys_per_head = []
        for i in range(self.num_waypoints - 1):
            xs_i = waypoint_samples[i]  # [b, dim]
            xs_i1 = waypoint_samples[i + 1]  # [b, dim]

            ones_labels = torch.ones(b, 1, device=self.device)
            zeros_labels = torch.zeros(b, 1, device=self.device)

            xs = torch.cat([xs_i, xs_i1], dim=0)  # [2*b, dim]
            ys = torch.cat([ones_labels, zeros_labels], dim=0)  # [2*b, 1]

            xs_per_head.append(xs)
            ys_per_head.append(ys)

        # build uniform predict_ldr MAE eval function if instrumentation is enabled.
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """compute MAE between predict_ldr(pstar_eval) and true_ldrs_eval.

                this is the uniform eval signal: single forward pass on eval pstar,
                sum logits across heads, measure error against reference LDRs. _model
                argument is ignored; self.predict_ldr closure-captures self for access
                to classifier logits.
                """
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        # train multi-head classifier
        self.classifier.fit(
            xs_per_head,
            ys_per_head,
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
            early_stop_cfg=self.early_stop_cfg,
            _meta_out=meta_out,
        )

        # extract training metadata
        self._final_step = meta_out.get("final_step", self.classifier.n_steps)
        self._stop_reason = meta_out.get("stop_reason", None)

    def predict_ldr(
        self,
        xs: torch.Tensor,  # [b, dim]
    ) -> torch.Tensor:
        logits = self.classifier.predict_logits(xs)  # [b, w-1]
        return logits.sum(dim=1)  # [b]
