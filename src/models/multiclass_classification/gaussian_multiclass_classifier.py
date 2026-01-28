import torch
import torch.nn as nn

from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier


class GaussianMulticlassClassifier(MulticlassClassifier):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_classes, input_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(num_classes, input_dim))
        self.c = nn.Parameter(torch.zeros(num_classes))
        self.num_classes = num_classes

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        quadratic_term = torch.einsum("bi,kij,bj->bk", xs, self.A, xs)
        linear_term = torch.einsum("bi,ki->bk", xs, self.b)
        return quadratic_term + linear_term + self.c

    def _reset_parameters(self) -> None:
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
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self.forward(xs)
            l = loss(y_pred, ys)
            l.backward()
            optimizer.step()
        if y_pred.isnan().any():
            breakpoint()

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(xs)
            return logits


def make_gaussian_multiclass_classifier(**kwargs) -> MulticlassClassifier:
    return GaussianMulticlassClassifier(**kwargs)
