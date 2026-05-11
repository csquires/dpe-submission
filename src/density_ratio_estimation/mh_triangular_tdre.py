import torch

from src.density_ratio_estimation.base import ELDR
from src.models.binary_classification.multi_head_binary_classifier import (
    MultiHeadBinaryClassifier,
)
from src.waypoints.triangular_waypoints import TriangularWaypointBuilder1D


class MultiHeadTriangularTDRE(ELDR):
    """
    TDRE variant using single multi-head binary classifier instead of list of classifiers.
    builds waypoints along triangular path p0 -> p* -> p1, then trains all heads simultaneously.
    """

    def __init__(
        self,
        classifier: MultiHeadBinaryClassifier,
        waypoint_builder: TriangularWaypointBuilder1D | None = None,
        num_waypoints: int = 5,
        device: str = "cuda",
    ) -> None:
        """
        Initialize MultiHeadTriangularTDRE.

        args:
            classifier: multi-head binary classifier with num_heads == num_waypoints - 1
            waypoint_builder: triangular waypoint builder (default: None, creates default)
            num_waypoints: number of waypoints in triangular path (default: 5)
            device: device to run classifier on (default: "cuda")
        """
        self.device = device
        self.num_waypoints = num_waypoints
        self.classifier = classifier.to(self.device)

        # validate head count
        if self.classifier.num_heads != num_waypoints - 1:
            raise ValueError(
                "MultiHeadBinaryClassifier must have num_heads == num_waypoints - 1"
            )

        # initialize waypoint builder with default if None
        if waypoint_builder is None:
            self.waypoint_builder = TriangularWaypointBuilder1D(
                midpoint_oversample=0,
                gamma_power=1.0,
                vertex=0.5,
            )
        else:
            self.waypoint_builder = waypoint_builder

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """
        Fit multi-head classifier on waypoint pairs.

        samples_p0: samples from p0, shape [n0, dim]
        samples_p1: samples from p1, shape [n1, dim]
        samples_pstar: samples from p*, shape [nstar, dim]
        """
        # build waypoints: [num_waypoints, batch_size, dim]
        waypoint_samples = self.waypoint_builder.build_waypoints(
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            num_waypoints=self.num_waypoints,
        )
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

        # train multi-head classifier
        self.classifier.fit(xs_per_head, ys_per_head)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict log density ratio by summing logits across heads.

        xs: sample points, shape [batch_size, dim]
        returns: ldr estimates, shape [batch_size]
        """
        # get multi-head logits: [batch_size, num_heads]
        logits = self.classifier.predict_logits(xs)

        # sum across heads: [batch_size]
        ldr = logits.sum(dim=1)

        return ldr

