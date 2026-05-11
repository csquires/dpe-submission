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
    ) -> None:
        self.device = device
        self.num_waypoints = num_waypoints
        self.classifier = classifier.to(self.device)
        self.waypoint_builder = waypoint_builder

        if self.classifier.num_heads != num_waypoints - 1:
            raise ValueError(
                "MultiHeadBinaryClassifier must have num_heads == num_waypoints - 1"
            )

    def fit(
        self,
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor,  # [b1, dim]
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(
            samples_p0, samples_p1, self.num_waypoints
        )  # [w, b, dim]
        b = waypoint_samples.shape[1]

        xs_per_head = []
        ys_per_head = []
        for i in range(self.num_waypoints - 1):
            xs = torch.cat([waypoint_samples[i], waypoint_samples[i + 1]], dim=0)
            p_num_labels = torch.ones((b, 1), dtype=torch.float, device=self.device)
            p_den_labels = torch.zeros((b, 1), dtype=torch.float, device=self.device)
            ys = torch.cat([p_num_labels, p_den_labels], dim=0)
            xs_per_head.append(xs)
            ys_per_head.append(ys)

        self.classifier.fit(xs_per_head, ys_per_head)

    def predict_ldr(
        self,
        xs: torch.Tensor,  # [b, dim]
    ) -> torch.Tensor:
        logits = self.classifier.predict_logits(xs)  # [b, w-1]
        return logits.sum(dim=1)  # [b]
