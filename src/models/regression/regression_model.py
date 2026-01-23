import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        raise NotImplementedError

    def predict(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DefaultRegressionModel(RegressionModel):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        ## TODO: implement training loop
        pass

    def predict(self, xs: torch.Tensor) -> torch.Tensor:
        return self.forward(xs)


def build_default_classifier(input_dim: int) -> nn.Module:
    return RegressionModel(DefaultRegressionModel(input_dim))