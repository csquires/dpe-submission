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