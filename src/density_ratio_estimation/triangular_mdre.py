import torch
from einops import rearrange

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier
from src.waypoints.triangular_waypoints import TriangularWaypointBuilder1D


class TriangularMDRE(DensityRatioEstimator):
    """
    MDRE variant that builds waymarks along a triangular path p0 -> p* -> p1.
    """
    def __init__(
        self,
        classifier: MulticlassClassifier,
        waypoint_builder: TriangularWaypointBuilder1D = None,
        # waypoint_builder: TriangularWaypointBuilder1D | None = None,
        device: str = "cuda",
        midpoint_oversample: int = 0,
        gamma_power: float = 1.0,
    ):
        self.device = device
        self.classifier = classifier.to(self.device)
        self.waypoint_builder = waypoint_builder or TriangularWaypointBuilder1D(
            midpoint_oversample=midpoint_oversample,
            gamma_power=gamma_power,
        )
        self.num_waypoints = self.classifier.num_classes

    def fit(
        self,
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor,  # [b1, dim]
        samples_pstar: torch.Tensor,  # [bstar, dim]
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            num_waypoints=self.num_waypoints,
        )  # [w, b, dim]
        b = waypoint_samples.shape[1]
        xs = rearrange(waypoint_samples, 'w b dim -> (w b) dim')
        ys = torch.cat([torch.ones(b, dtype=torch.long) * i for i in range(self.num_waypoints)]).to(self.device)
        self.classifier.fit(xs, ys)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        logits = self.classifier.predict_logits(xs)
        p1_logits = logits[:, -1]
        p0_logits = logits[:, 0]
        return p0_logits - p1_logits
