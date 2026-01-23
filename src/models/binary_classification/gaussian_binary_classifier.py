import torch
import torch.nn as nn

from src.models.binary_classification.binary_classifier import BinaryClassifier


class GaussianBinaryClassifier(BinaryClassifier):
    def __init__(
        self, 
        input_dim: int
    ):
        super().__init__()
        self.A = nn.Parameter(torch.randn(input_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        quadratic_term = torch.einsum("bi,ij,bj->b", xs, self.A, xs)
        return (quadratic_term + xs @ self.b + self.c).unsqueeze(1)

    def _reset_parameters(self) -> None:
        """Reset model parameters to random initialization."""
        nn.init.normal_(self.A, mean=0, std=1.0)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.c)

    def fit(
        self, 
        xs: torch.Tensor, 
        ys: torch.Tensor,
        learning_rate: float = 0.05,
        num_epochs: int = 1000,
    ) -> None:
        self._reset_parameters()
        self.train()
        loss = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self.forward(xs)
            l = loss(y_pred, ys)
            l.backward()
            print(f"Loss: {l.item()}")
            optimizer.step()
        if y_pred.isnan().any():
            breakpoint()

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(xs)
            return logits.squeeze(1)


def build_gaussian_binary_classifier(
    input_dim: int, 
) -> BinaryClassifier:
    return GaussianBinaryClassifier(input_dim)