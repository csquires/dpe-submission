from typing import Any, Callable

import torch
import torch.nn as nn

from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier
from src.methods.common._report import _make_report


class DefaultMulticlassClassifier(MulticlassClassifier):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        # model hyperparameters
        latent_dim: int = 10,
        n_hidden_layers: int = 2,
        # training hyperparameters
        learning_rate: float = 0.05,
        num_epochs: int = 1000,
        batch_size: int | None = None,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        layers = []
        layers.append(nn.Linear(input_dim, latent_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent_dim, num_classes))
        self.model = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

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
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_fn: Callable[[Any], torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """train via AdamW + CrossEntropyLoss.

        falls back to full-batch when self.batch_size is None or >= n.
        if step_cb and eval_fn are supplied, _make_report fires the callback
        every step_cb_interval optimizer updates -- same contract as
        DefaultBinaryClassifier.fit.
        """
        self._reset_parameters()
        self.train()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        n = xs.shape[0]
        bs = self.batch_size if (self.batch_size and self.batch_size < n) else n
        do_report = _make_report(step_cb, step_cb_interval, eval_fn, self, self)
        for epoch in range(self.num_epochs):
            if bs == n:
                optimizer.zero_grad()
                y_pred = self.forward(xs)
                l = loss(y_pred, ys)
                l.backward()
                optimizer.step()
                do_report()
            else:
                perm = torch.randperm(n, device=xs.device)
                for start in range(0, n, bs):
                    idx = perm[start:start + bs]
                    optimizer.zero_grad()
                    y_pred = self.forward(xs[idx])
                    l = loss(y_pred, ys[idx])
                    l.backward()
                    optimizer.step()
                    do_report()
        if y_pred.isnan().any():
            breakpoint()

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(xs)
            return logits.squeeze(1)


def make_default_multiclass_classifier(**kwargs) -> MulticlassClassifier:
    return DefaultMulticlassClassifier(**kwargs)