from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BinaryClassifier(nn.Module, ABC):
    @abstractmethod
    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DefaultBinaryClassifier(BinaryClassifier):
    def __init__(
        self, 
        input_dim: int, 
        # model hyperparameters
        latent_dim: int = 10,
        num_layers: int = 2,
        # training hyperparameters
        num_epochs: int = 100,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _reset_parameters(self) -> None:
        """Reset model parameters to random initialization."""
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        self._reset_parameters()
        self.train()
        loss = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
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
            return self.forward(xs)


def build_default_binary_classifier(
    input_dim: int, 
    latent_dim: int = 10,
    num_layers: int = 2,    
    num_epochs: int = 100,
    learning_rate: float = 0.01,
) -> BinaryClassifier:
    return DefaultBinaryClassifier(input_dim, latent_dim, num_layers, num_epochs, learning_rate)