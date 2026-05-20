from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn


class MulticlassClassifier(nn.Module, ABC):
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
        """train the multiclass classifier on labeled data.

        same step_cb / eval_fn / step_cb_interval contract as
        src.models.binary_classification.BinaryClassifier.fit. step_cb
        receives (step: int, score: float); score is eval_fn(model).item()
        evaluated under torch.no_grad() with the model toggled to eval.
        step_cb_interval is the number of training updates between
        callback invocations.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
