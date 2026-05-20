from typing import Callable

import torch
from einops import rearrange

from ...common.base import DRE
from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier
from src.waypoints.waypoints1d import WaypointBuilder1D, DefaultWaypointBuilder1D


class MDRE(DRE):
    def __init__(
        self,
        classifier: MulticlassClassifier,
        waypoint_builder: WaypointBuilder1D = DefaultWaypointBuilder1D(),
        device: str = "cuda"
    ):
        self.device = device
        self.classifier = classifier.to(self.device)
        self.waypoint_builder = waypoint_builder
        self.num_waypoints = self.classifier.num_classes

    def fit(
        self,
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor,   # [b1, dim]
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """build waymark samples and train classifier with optional pruning.

        step_cb / eval_data / step_cb_interval mirror BDRE.fit. when both
        step_cb and eval_data are supplied, eval_fn closes over predict_ldr
        evaluated on eval_data["pstar"] and returns MAE against
        eval_data["true_ldrs"]; otherwise eval_fn is None and no callbacks
        fire.
        """
        waypoint_samples = self.waypoint_builder.build_waypoints(samples_p0, samples_p1, self.num_waypoints)  # [w, b, dim]
        b = waypoint_samples.shape[1]
        xs = rearrange(waypoint_samples, 'w b dim -> (w b) dim')  # [n, dim] for n = w * b
        ys = torch.cat([torch.ones(b, dtype=torch.long) * i for i in range(self.num_waypoints)]).to(self.device)  # [n]

        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """held-out MAE between predict_ldr(pstar) and true_ldrs."""
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        self.classifier.fit(
            xs,
            ys,
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
        )

    def predict_ldr(
        self,
        xs: torch.Tensor  # [b, dim]
    ) -> torch.Tensor:
        logits = self.classifier.predict_logits(xs)
        p1_logits = logits[:, -1]
        p0_logits = logits[:, 0]
        return p0_logits - p1_logits
