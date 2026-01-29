import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.binary_classification.binary_classifier import BinaryClassifier
from src.waypoints.triangular_waypoints import TriangularWaypointBuilder1D


class TriangularTDRE(DensityRatioEstimator):
    """
    TDRE variant that builds waymarks along a triangular path p0 -> p* -> p1.
    """
    def __init__(
        self,
        classifiers: list[BinaryClassifier],
        waypoint_builder: TriangularWaypointBuilder1D = None,
        # waypoint_builder: TriangularWaypointBuilder1D | None = None,
        num_waypoints: int = 5,
        device: str = "cuda",
        midpoint_oversample: int = 0,
        gamma_power: float = 1.0,
    ):
        self.device = device
        self.classifiers = [classifier.to(self.device) for classifier in classifiers]
        self.waypoint_builder = waypoint_builder or TriangularWaypointBuilder1D(
            midpoint_oversample=midpoint_oversample,
            gamma_power=gamma_power,
        )
        self.num_waypoints = num_waypoints

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            num_waypoints=self.num_waypoints,
        )
        b = waypoint_samples.shape[1]
        for i in range(self.num_waypoints - 1):
            xs = torch.cat([waypoint_samples[i], waypoint_samples[i + 1]], dim=0)
            p_num_labels = torch.ones((b, 1), dtype=torch.float, device=self.device)
            p_den_labels = torch.zeros((b, 1), dtype=torch.float, device=self.device)
            ys = torch.cat([p_num_labels, p_den_labels], dim=0)
            self.classifiers[i].fit(xs, ys)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        waypoint_ldrs = torch.zeros(xs.shape[0], self.num_waypoints - 1, device=self.device)
        for i in range(self.num_waypoints - 1):
            waypoint_ldrs[:, i] = self.classifiers[i].predict_logits(xs)
        return waypoint_ldrs.sum(axis=1)
