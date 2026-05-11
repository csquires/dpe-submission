import torch
from einops import rearrange

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier
from src.waypoints.waypoints1d import WaypointBuilder1D, DefaultWaypointBuilder1D


class MDRE(DensityRatioEstimator):
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
        samples_p1: torch.Tensor   # [b1, dim]
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(samples_p0, samples_p1, self.num_waypoints)  # [w, b, dim]
        b = waypoint_samples.shape[1]
        xs = rearrange(waypoint_samples, 'w b dim -> (w b) dim')  # [n, dim] for n = w * b
        ys = torch.cat([torch.ones(b, dtype=torch.long) * i for i in range(self.num_waypoints)]).to(self.device)  # [n]
        self.classifier.fit(xs, ys)

    def predict_ldr(
        self, 
        xs: torch.Tensor  # [b, dim]
    ) -> torch.Tensor:
        logits = self.classifier.predict_logits(xs)
        p1_logits = logits[:, -1]
        p0_logits = logits[:, 0]
        return p0_logits - p1_logits

