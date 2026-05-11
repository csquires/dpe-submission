import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "src.methods.cls.tdre is deprecated and will be removed "
    "in a future cycle. Migration: use src.methods.cls.tdre.mh.MultiHeadTDRE "
    "(TDRE with separate per-pair classifiers is deprecated).",
    DeprecationWarning,
    stacklevel=2,
)

import torch

from ...common.base import DRE
from src.models.binary_classification.binary_classifier import BinaryClassifier
from src.waypoints.waypoints1d import WaypointBuilder1D, DefaultWaypointBuilder1D


class TDRE(DRE):
    def __init__(
        self, 
        classifiers: list[BinaryClassifier],
        waypoint_builder: WaypointBuilder1D = DefaultWaypointBuilder1D(),
        num_waypoints: int = 10,
        device: str = "cuda"
    ):
        # note: the i-th classifier discrimates between waypoint i (in the numerator) and waypoint i+1 (in the denominator)
        self.device = device
        self.classifiers = [classifier.to(self.device) for classifier in classifiers]
        self.waypoint_builder = waypoint_builder
        self.num_waypoints = num_waypoints
        
    def fit(
        self, 
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor   # [b1, dim]
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(samples_p0, samples_p1, self.num_waypoints)  # [w, b, dim]
        b = waypoint_samples.shape[1]
        for i in range(self.num_waypoints - 1):
            xs = torch.cat([waypoint_samples[i], waypoint_samples[i+1]], dim=0)
            p_num_labels = torch.ones((b, 1), dtype=torch.float).to(self.device)
            p_den_labels = torch.zeros((b, 1), dtype=torch.float).to(self.device)
            ys = torch.cat([p_num_labels, p_den_labels], dim=0).to(self.device)
            self.classifiers[i].fit(xs, ys)

    def predict_ldr(
        self, 
        xs: torch.Tensor  # [b, dim]
    ) -> torch.Tensor:
        waypoint_ldrs = torch.zeros(xs.shape[0], self.num_waypoints-1).to(self.device)  # [b, w-1]
        for i in range(self.num_waypoints-1):
            waypoint_ldrs[:, i] = self.classifiers[i].predict_logits(xs)
        return waypoint_ldrs.sum(axis=1)  # [b]

