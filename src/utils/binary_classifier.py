import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        raise NotImplementedError

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DefaultClassifier(BinaryClassifier):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        ## TODO: implement trainin loop
        pass

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        return self.forward(xs)


def build_default_classifier(input_dim: int) -> nn.Module:
    return BinaryClassifier(DefaultClassifier(input_dim))