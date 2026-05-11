import torch

from ...common.base import DRE
from src.models.binary_classification.binary_classifier import BinaryClassifier


class BDRE(DRE):
    def __init__(
        self,
        classifier: BinaryClassifier,
        device: str = "cuda"
    ):
        # extract input_dim from classifier's first layer
        if hasattr(classifier, 'input_dim'):
            input_dim = classifier.input_dim
        elif hasattr(classifier, 'model') and len(classifier.model) > 0:
            # DefaultBinaryClassifier: first layer is nn.Linear
            input_dim = classifier.model[0].in_features
        elif hasattr(classifier, 'backbone') and len(classifier.backbone) > 0:
            # MultiHeadBinaryClassifier: first layer of backbone is nn.Linear
            input_dim = classifier.backbone[0].in_features
        elif hasattr(classifier, 'A'):
            # GaussianBinaryClassifier: A is [input_dim, input_dim]
            input_dim = classifier.A.shape[0]
        else:
            raise ValueError(
                f"Cannot determine input_dim from {type(classifier).__name__}. "
                "Classifier must have 'input_dim' attribute or detectable first layer."
            )

        super().__init__(input_dim)
        self.device = device
        self.classifier = classifier.to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        xs = torch.cat([samples_p0, samples_p1], dim=0)
        p0_labels = torch.ones((samples_p0.shape[0], 1), dtype=torch.float).to(self.device)
        p1_labels = torch.zeros((samples_p1.shape[0], 1), dtype=torch.float).to(self.device)
        ys = torch.cat([p0_labels, p1_labels], dim=0)
        self.classifier.fit(xs, ys)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        return self.classifier.predict_logits(xs)



