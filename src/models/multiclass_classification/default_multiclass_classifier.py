import torch
import torch.nn as nn

from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier


class DefaultMulticlassClassifier(MulticlassClassifier):
    def __init__(
        self, 
        input_dim: int,
        num_classes: int,
        # model hyperparameters
        latent_dim: int = 10,
        num_layers: int = 3,
        # training hyperparameters
        learning_rate: float = 0.05,
        num_epochs: int = 1000,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, latent_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent_dim, num_classes))
        self.model = nn.Sequential(*layers)
        self.num_classes = num_classes
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

    def fit(
        self, 
        xs: torch.Tensor,  # [n, dim]
        ys: torch.Tensor,  # [n], with values in {0, ..., num_classes-1}
    ) -> None:
        self._reset_parameters()
        self.train()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
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
            logits = self.forward(xs)
            return logits.squeeze(1)


def make_default_multiclass_classifier(**kwargs) -> MulticlassClassifier:
    return DefaultMulticlassClassifier(**kwargs)