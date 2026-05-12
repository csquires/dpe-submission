from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn


class BinaryClassifier(nn.Module, ABC):
    @abstractmethod
    def fit(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_fn: Callable[[Any], torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """
        Train the binary classifier on labeled data.

        Fit xs and ys via SGD; when step_cb and eval_fn are provided, invoke
        pruning callbacks at step_cb_interval granularity for optuna trial control.

        Args:
            xs: input features, shape (n_samples, n_features).
            ys: binary labels, shape (n_samples,).
            step_cb: optional callback of signature (step: int, score: float) -> None,
                invoked every step_cb_interval training steps. score is a scalar
                extracted from eval_fn(model).item(). When None, no callbacks are issued.
            eval_fn: optional callable that accepts the model and returns a 0-dimensional
                torch.Tensor (scalar metric). Invoked every step_cb_interval steps inside
                a no_grad context. Common choice is a closure over held-out eval data.
            step_cb_interval: number of minibatch SGD updates between callback invocations.
                Default 50. Ignored if step_cb is None.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError