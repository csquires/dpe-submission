import math
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


def logmeanexp(z: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    log E[exp(z)] over the given dimension.

    logsumexp(z, dim) - log(z.shape[dim]). numerically stable via logsumexp.
    edge case: dim size 0 -> logsumexp returns -inf; log(0) = -inf. result is -inf.

    z: [..., N, ...] where N is at position dim.
    returns: [...] with dimension dim removed.
    """
    return torch.logsumexp(z, dim=dim) - math.log(z.shape[dim])


class LossSpec(ABC):
    """
    abc for loss functions and ldr prediction. partitions logits/labels by
    P (label=1, numerator) vs Q (label=0, denominator).
    """

    @abstractmethod
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        compute scalar loss from logits and binary group labels.

        logits: [N], float32. unclipped critic output.
        labels: [N], long or float32 (0 or 1). P=1, Q=0.
        returns: scalar loss (minimized via backward).

        edge case: empty P or Q -> mean() returns nan -> loss is nan.
        """

    @abstractmethod
    def predict_ldr(self, logits: torch.Tensor) -> torch.Tensor:
        """
        predict log-density-ratio from critic logits.

        logits: [N], float32.
        returns: [N] log-density-ratio estimates.
        """

    def finalize(self, denom_logits: torch.Tensor) -> None:
        """
        optional normalization pass (called after each rung eval).
        no-op in BCE/NWJ. DV overrides to update self.c.

        denom_logits: [M], float32. denominator group logits.
        """
        pass


class BCESpec(LossSpec):
    """
    bce with logits. bayes-optimal logit is the true log-density-ratio.
    """

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """bce with logits: numerically stable fusion of sigmoid + cross-entropy."""
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def predict_ldr(self, logits: torch.Tensor) -> torch.Tensor:
        """return logits directly."""
        return logits


class NWJSpec(LossSpec):
    """
    nowozin-welling-jitkrittum ratio. loss = -mean(T[P]) + mean(exp(T[Q]-1)).
    predict_ldr = logits - 1. no clamp before exp (zero-init in critics.py enforces).
    """

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        partition on labels, compute -mean(T[P]) + mean(exp(T[Q]-1)).

        edge cases: empty P or Q -> nan.
        """
        T_P = logits[labels == 1]
        T_Q = logits[labels == 0]
        return -T_P.mean() + (T_Q - 1).exp().mean()

    def predict_ldr(self, logits: torch.Tensor) -> torch.Tensor:
        """return logits - 1."""
        return logits - 1


class DVSpec(LossSpec):
    """
    donsker-varadhan ratio. loss = -mean(T[P]) + logmeanexp(T[Q]).
    predict_ldr = logits - self.c. c is normalized per rung via finalize.
    """

    def __init__(self, mine_ema: bool = False, clamp: float | None = 10.0):
        """
        mine_ema: if True, apply mine-style ema debiasing to logmeanexp term.
        clamp: bound |T| before the logmeanexp/exp so a few outlier critic
            values on distant anchors cannot blow up the DV constant c. only
            bites when |T| > clamp (well-conditioned anchors like q0 are
            unaffected). None disables. default 10.0.

        self.c: normalization constant, initialized to 0.
        self.c_ema_old: prior value for ema (if enabled).
        """
        self.mine_ema = mine_ema
        self.clamp = clamp
        self.c = 0.0
        self.c_ema_old = 0.0

    def _lme(self, z: torch.Tensor) -> torch.Tensor:
        """clamped logmeanexp: bounds e^T so distant-anchor outliers cannot
        dominate the denominator (the q1 blow-up)."""
        if self.clamp is not None:
            z = z.clamp(-self.clamp, self.clamp)
        return logmeanexp(z, dim=0)

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        partition on labels, compute -mean(T[P]) + logmeanexp(T[Q]) (clamped).
        if mine_ema: blend denominator with c_ema_old (coeff=0.01).

        edge cases: empty P -> nan; empty Q -> logmeanexp returns -inf.
        """
        T_P = logits[labels == 1]
        T_Q = logits[labels == 0]

        # bounded-critic DV: clamp T on BOTH groups. DV's logmeanexp only weakly
        # penalizes large T (unlike NWJ's exp), so -mean(T_P) drives T unbounded
        # on distant anchors. clamping the numerator too caps that drift.
        if self.clamp is not None:
            T_P = T_P.clamp(-self.clamp, self.clamp)
        numerator = -T_P.mean()
        denominator = self._lme(T_Q)

        if self.mine_ema:
            ema_coeff = 0.01
            denominator = ema_coeff * self.c_ema_old + (1 - ema_coeff) * denominator

        return numerator + denominator

    def predict_ldr(self, logits: torch.Tensor) -> torch.Tensor:
        """clamp T consistently with the loss/constant, then subtract c."""
        t = logits.clamp(-self.clamp, self.clamp) if self.clamp is not None else logits
        return t - self.c

    def finalize(self, denom_logits: torch.Tensor) -> None:
        """
        recompute c under no_grad from the denominator draw (clamped logmeanexp).
        updates self.c and self.c_ema_old (for next finalize).

        edge case: denom_logits is [0] -> logmeanexp returns -inf. self.c = -inf.
          predict_ldr will be logits + inf (degenerate; guard at adapter level).
        """
        with torch.no_grad():
            c_new = self._lme(denom_logits).item()
            if self.mine_ema:
                ema_coeff = 0.01
                self.c = ema_coeff * self.c_ema_old + (1 - ema_coeff) * c_new
            else:
                self.c = c_new
            self.c_ema_old = c_new
