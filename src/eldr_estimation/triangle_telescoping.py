from typing import Callable

import torch

from src.models.binary_classification.binary_classifier import BinaryClassifier
from src.models.binary_classification.default_binary_classifier import build_default_binary_classifier
from src.eldr_estimation.base import ELDREstimator
from src.waypoints.waypoints2d import WaypointBuilder2D, DefaultWaypointBuilder2D


class TriangleTelescopingEstimator(ELDREstimator):
    def __init__(
        self, 
        input_dim: int,
        classifier_builder: Callable[[], BinaryClassifier] = build_default_binary_classifier,
        waypoint_builder: WaypointBuilder2D = DefaultWaypointBuilder2D(),
        device: str = "cuda"
    ):
        self.input_dim = input_dim
        self.classifier = classifier_builder(input_dim).to(device)
        self.waypoint_builder = waypoint_builder
        self.device = device

    def _fit_classifiers(
        self,
        samples_pstar: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor
    ):
        waypoint_samples = self.waypoint_builder.build_waypoints(samples_pstar, samples_p0, samples_p1)
        b = waypoint_samples.shape[1]
        for i in range(self.num_waypoints - 1):
            xs = torch.cat([waypoint_samples[i], waypoint_samples[i+1]], dim=0)
            p_num_labels = torch.ones((b, 1), dtype=torch.float).to(self.device)
            p_den_labels = torch.zeros((b, 1), dtype=torch.float).to(self.device)
            ys = torch.cat([p_num_labels, p_den_labels], dim=0).to(self.device)
            self.classifiers[i].fit(xs, ys)

    def _predict_ldrs(
        self,
        samples_pstar: torch.Tensor,
    ):
        waypoint_ldrs = torch.zeros(samples_pstar.shape[0], self.num_waypoints-1).to(self.device)  # [b, w-1]
        for i in range(self.num_waypoints-1):
            waypoint_ldrs[:, i] = self.classifiers[i].predict_logits(samples_pstar)
        return waypoint_ldrs.sum(axis=1)  # [b]
        
    def estimate_eldr(
        self, 
        samples_pstar: torch.Tensor, 
        samples_p0: torch.Tensor, 
        samples_p1: torch.Tensor
    ) -> float:
        # TODO: we might want to add in crossfitting
        self._fit_classifiers(samples_pstar, samples_p0, samples_p1)
        est_ldrs = self.classifier.predict_logits(samples_pstar)
        return torch.mean(est_ldrs)