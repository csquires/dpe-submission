from typing import Callable

import torch

from ...common.base import DRE
from src.models.binary_classification.binary_classifier import BinaryClassifier


class BDRE(DRE):
    def __init__(
        self,
        classifier: BinaryClassifier,
        device: str = "cuda",
        early_stop_cfg: dict | None = None
    ):
        # extract input_dim from classifier's first layer
        if hasattr(classifier, 'input_dim'):
            input_dim = classifier.input_dim
        elif hasattr(classifier, 'model') and len(classifier.model) > 0:
            # DefaultBinaryClassifier: first layer is nn.Linear
            input_dim = classifier.model[0].in_features
        elif hasattr(classifier, 'backbone') and len(classifier.backbone) > 0:
            # MultiHeadBinaryClassifier: first layer of backbone is nn.Linear
            input_dim = classifier.backbone[0].in_features
        elif hasattr(classifier, 'A'):
            # GaussianBinaryClassifier: A is [input_dim, input_dim]
            input_dim = classifier.A.shape[0]
        else:
            raise ValueError(
                f"Cannot determine input_dim from {type(classifier).__name__}. "
                "Classifier must have 'input_dim' attribute or detectable first layer."
            )

        super().__init__(input_dim)
        self.device = device
        self.classifier = classifier.to(self.device)
        self.early_stop_cfg = early_stop_cfg

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        xs = torch.cat([samples_p0, samples_p1], dim=0)
        p0_labels = torch.ones((samples_p0.shape[0], 1), dtype=torch.float).to(self.device)
        p1_labels = torch.zeros((samples_p1.shape[0], 1), dtype=torch.float).to(self.device)
        ys = torch.cat([p0_labels, p1_labels], dim=0)

        # build eval_fn closure if both step_cb and eval_data are provided
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """compute MAE between predict_ldr output and held-out ground truth.

                _model arg is part of the shared contract but ignored here; the closure
                captures self for access to predict_ldr. returned tensor is a 0-dim scalar.
                """
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        meta_out: dict = {}
        self.classifier.fit(
            xs,
            ys,
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
            early_stop_cfg=self.early_stop_cfg,
            _meta_out=meta_out,
        )
        self._final_step = meta_out.get("final_step", self.n_steps)
        self._stop_reason = meta_out.get("stop_reason", None)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        return self.classifier.predict_logits(xs)



