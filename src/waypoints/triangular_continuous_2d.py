"""V3 stacked-interpolant 2D-time triangular CTSM path concrete class.

Implements Stacked2DCtsm(CtsmPath2D) for V3 of TriangularCTSM. The interpolant
stacks two linear mixes:
  I_1 = (1 - t_1) x_0 + t_1 x_1
  \\mu = (1 - t_2) I_1 + t_2 x_*
and adds Gaussian noise with variance \\sigma^2 g(t_1, t_2). Two noise schedules
are supported via string dispatch:
  sqrt:         g = t_1 (1 - t_1)(1 - t_2)
  linear-stiff: g = (1 - exp(-k t_1))(1 - exp(-k(1 - t_1)))   (independent of t_2)

The closed-form 2-vector regression target for component i (i in {1, 2})
matches the conditional Gaussian time-score \\partial_{t_i} \\log \\rho_c. Two
formulations are kept inline-commented for A/B testing:
  Option A2 (active):    target_i = sigma^2 dg_dt_i (|eps|^2 - d)/2 + sigma sqrt(g) (dmu_dt_i . eps)
                         lambda_t_i = sigma^2 g  (uniform weighting)
  Option A1 (commented): same target divided by sqrt(2 |dmu_dt_i|^2 + tiny)
                         lambda_t_i = sigma^2 g / sqrt(2 |dmu_dt_i|^2 + tiny)
Both yield target_i / lambda_t_i = \\partial_{t_i} \\log \\rho_c.

Bug fix from bak: in linear-stiff schedule, _dgamma_dt2 returns zeros (bak
returned -gamma, which is incorrect since g has no t_2 dependence in this
schedule).

t2_max is a hyperparameter read by the estimator (TriangularCTSM2D) to clamp
training t_2 samples; not used inside this class.
"""
import torch
from torch import Tensor

from src.waypoints.path_2d import CtsmPath2D, VfmPath2D


class Stacked2DCtsm(CtsmPath2D):
    """Stacked-interpolant 2D-time CTSM path with closed-form 2-vector target.

    Parameterizes a continuous surface over (t_1, t_2) in [eps, 1-eps]^2 from p_0
    (at t = (0, 0)) through a p_* mixture toward p_1 (at t = (1, 0)) via a stacked
    linear interpolant plus Gaussian noise with variance sigma^2 g(t_1, t_2).

    Two target formulas are inline-commented in sample_and_target:
      Option A2 (uniform weighting, currently ACTIVE).
      Option A1 (drift-magnitude normalization, COMMENTED for A/B).

    Constructor parameters:
      sigma: scalar > 0, noise scale.
      gamma_schedule: "sqrt" or "linear-stiff" — selects noise schedule g(t_1, t_2).
      k: scalar > 0, stiffness for linear-stiff schedule (ignored for sqrt).
      t2_max: scalar in (0, 1), upper bound on training t_2. Read by the estimator
              to restrict training distribution; not used inside this class.
      eps: scalar > 0, boundary epsilon. Passed to Path2D.__init__.

    Inherits self.eps from Path2D.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        gamma_schedule: str = "sqrt",  # "sqrt" or "linear-stiff"
        k: float = 24.0,
        t2_max: float = 0.3,
        eps: float = 1e-3,
    ) -> None:
        """Initialize stacked 2D-time triangular CTSM path.

        Args:
            sigma: noise scale, must be > 0.
            gamma_schedule: noise schedule selector, "sqrt" or "linear-stiff".
            k: stiffness parameter for linear-stiff schedule, must be > 0.
            t2_max: upper bound on t_2 for training, must be in (0, 1).
            eps: boundary epsilon, passed to Path2D.__init__.

        Raises:
            ValueError: if sigma <= 0, gamma_schedule not in allowed set,
                        k <= 0, or t2_max not in (0, 1).
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if gamma_schedule not in {"sqrt", "linear-stiff"}:
            raise ValueError(
                f"gamma_schedule must be 'sqrt' or 'linear-stiff', got {gamma_schedule!r}"
            )
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if not (0.0 < t2_max < 1.0):
            raise ValueError(f"t2_max must be in (0, 1), got {t2_max}")

        self.sigma = sigma
        self.gamma_schedule = gamma_schedule
        self.k = k
        self.t2_max = t2_max
        super().__init__(eps)

    def _gamma(self, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute g(t_1, t_2). Output shape [B, 1]."""
        if self.gamma_schedule == "sqrt":
            return t1 * (1.0 - t1) * (1.0 - t2)
        elif self.gamma_schedule == "linear-stiff":
            # ignores t_2
            e1 = torch.exp(-self.k * t1)
            e2 = torch.exp(-self.k * (1.0 - t1))
            return (1.0 - e1) * (1.0 - e2)
        else:
            raise ValueError(f"unknown gamma_schedule: {self.gamma_schedule!r}")

    def _dgamma_dt1(self, t1: Tensor, t2: Tensor, g: Tensor) -> Tensor:
        """Compute partial g / partial t_1. Output shape [B, 1]. g is passed for reuse."""
        if self.gamma_schedule == "sqrt":
            return (1.0 - 2.0 * t1) * (1.0 - t2)
        elif self.gamma_schedule == "linear-stiff":
            e1 = torch.exp(-self.k * t1)
            e2 = torch.exp(-self.k * (1.0 - t1))
            # d/dt1 [(1 - e1)(1 - e2)] = k e1 (1 - e2) - k e2 (1 - e1)
            return self.k * e1 * (1.0 - e2) - self.k * e2 * (1.0 - e1)
        else:
            raise ValueError(f"unknown gamma_schedule: {self.gamma_schedule!r}")

    def _dgamma_dt2(self, t1: Tensor, t2: Tensor, g: Tensor) -> Tensor:
        """Compute partial g / partial t_2. Output shape [B, 1].

        BAK BUG FIX: bak returned -gamma for linear-stiff. Correct value is zero,
        since linear-stiff g has no t_2 dependence.
        """
        if self.gamma_schedule == "sqrt":
            return -t1 * (1.0 - t1)
        elif self.gamma_schedule == "linear-stiff":
            return torch.zeros_like(g)  # bak bug fix: was `-gamma`. correct value is 0.
        else:
            raise ValueError(f"unknown gamma_schedule: {self.gamma_schedule!r}")

    def sample_and_target(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        t1: Tensor,        # [B, 1] in [self.eps, 1 - self.eps]
        t2: Tensor,        # [B, 1] in [self.eps, self.t2_max]
        epsilon: Tensor,   # [B, D]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample stacked interpolant and return closed-form 2-vector target.

        See module docstring for math. Returns (x, target [B, 2], lambda_t [B, 2]).
        target and lambda_t are detached; x is NOT detached.
        """
        # stacked interpolant
        i_1 = (1.0 - t1) * x0 + t1 * x1                 # [B, D]
        mu  = (1.0 - t2) * i_1 + t2 * xstar             # [B, D]

        # noise schedule and std
        g    = self._gamma(t1, t2)                      # [B, 1]
        std  = self.sigma * torch.sqrt(g)               # [B, 1]

        # noisy sample on path
        x    = mu + std * epsilon                       # [B, D]

        # partial derivatives of mu w.r.t. t_1, t_2
        dmu_dt1 = (1.0 - t2) * (x1 - x0)                # [B, D]
        dmu_dt2 = xstar - i_1                           # [B, D]

        # partial derivatives of g w.r.t. t_1, t_2
        dg_dt1 = self._dgamma_dt1(t1, t2, g)            # [B, 1]
        dg_dt2 = self._dgamma_dt2(t1, t2, g)            # [B, 1]

        # quantities reused in target
        eps_sq        = (epsilon ** 2).sum(dim=-1, keepdim=True)         # [B, 1]
        dim           = epsilon.shape[-1]
        dmu_dt1_dot_e = (dmu_dt1 * epsilon).sum(dim=-1, keepdim=True)    # [B, 1]
        dmu_dt2_dot_e = (dmu_dt2 * epsilon).sum(dim=-1, keepdim=True)    # [B, 1]
        sigma_sq      = self.sigma ** 2

        # ============================================================
        # OPTION A1 (drift-magnitude normalized) — COMMENTED for A/B:
        # tiny    = 1e-8
        # denom_1 = torch.sqrt(2.0 * (dmu_dt1 ** 2).sum(dim=-1, keepdim=True) + tiny)  # [B, 1]
        # denom_2 = torch.sqrt(2.0 * (dmu_dt2 ** 2).sum(dim=-1, keepdim=True) + tiny)  # [B, 1]
        # target_1 = (sigma_sq * dg_dt1 * (eps_sq - dim) / 2.0 + std * dmu_dt1_dot_e) / denom_1
        # target_2 = (sigma_sq * dg_dt2 * (eps_sq - dim) / 2.0 + std * dmu_dt2_dot_e) / denom_2
        # lambda_t_1 = sigma_sq * g / denom_1
        # lambda_t_2 = sigma_sq * g / denom_2
        # target   = torch.cat([target_1, target_2], dim=-1)
        # lambda_t = torch.cat([lambda_t_1, lambda_t_2], dim=-1)
        # ============================================================

        # OPTION A2 (uniform weighting) — currently ACTIVE:
        target_1 = sigma_sq * dg_dt1 * (eps_sq - dim) / 2.0 + std * dmu_dt1_dot_e   # [B, 1]
        target_2 = sigma_sq * dg_dt2 * (eps_sq - dim) / 2.0 + std * dmu_dt2_dot_e   # [B, 1]
        target   = torch.cat([target_1, target_2], dim=-1)                          # [B, 2]
        lam      = sigma_sq * g                                                     # [B, 1]
        lambda_t = lam.expand(-1, 2)                                                # [B, 2]

        return x, target.detach(), lambda_t.detach()


class Stacked2DVfm(VfmPath2D):
    """Stacked-interpolant 2D-time VFM path with linear-stiff noise amplitude.

    Parameterizes a continuous surface over (t_1, t_2) in [eps, 1-eps] x [eps, t2_max]
    from p_0 (at t=(0,0)) through p_* toward p_1 (at t=(1,0)) via stacked linear
    interpolant plus Gaussian noise with amplitude gamma(t_1, t_2) (NOT variance).

    Interpolant: mu = (1 - t_2)((1 - t_1) x_0 + t_1 x_1) + t_2 x_*

    Noise amplitude gamma schedule selectable via gamma_schedule string-dispatch.
    Currently only "linear-stiff" implemented; reserves "sqrt" slot for future ablation.
    Under linear-stiff, gamma is independent of t_2, so dgamma_dt2 = 0 exactly.

    Constructor parameters:
      k: scalar > 0, stiffness for linear-stiff noise amplitude schedule.
      gamma_schedule: "linear-stiff" only; string-dispatch slot for future expansions.
      t2_max: scalar in (0, 1), upper bound on training t_2. Read by estimator.
      eps: scalar >= 1e-3, boundary epsilon. Passed to VfmPath2D.__init__.

    Naming convention: gamma here is the noise AMPLITUDE (std), not variance.
    This matches BarycentricVfm1D, contrasting with Stacked2DCtsm.sigma^2*g.
    """

    def __init__(
        self,
        k: float = 20.0,
        gamma_schedule: str = "linear-stiff",
        t2_max: float = 0.3,
        eps: float = 1e-3,
    ) -> None:
        """Initialize stacked 2D-time triangular VFM path.

        Args:
            k: stiffness parameter for linear-stiff noise amplitude, must be > 0.
            gamma_schedule: noise amplitude schedule selector. Only "linear-stiff"
                           implemented; "sqrt" reserved for future ablation.
            t2_max: upper bound on t_2 for training, must be in (0, 1).
            eps: boundary epsilon, must be >= 1e-3 (boundary regularity floor).
                 Passed to VfmPath2D.__init__.

        Raises:
            ValueError: if k <= 0, t2_max not in (0, 1), or eps < 1e-3.
            NotImplementedError: if gamma_schedule not in {"linear-stiff"}.
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if eps < 1e-3:
            raise ValueError(f"eps must be >= 1e-3 (boundary regularity floor), got {eps}")
        if not (0.0 < t2_max < 1.0):
            raise ValueError(f"t2_max must be in (0, 1), got {t2_max}")
        if gamma_schedule != "linear-stiff":
            raise NotImplementedError(
                f"gamma_schedule={gamma_schedule!r} not implemented; "
                "only 'linear-stiff' is supported in this PR. "
                "The string-dispatch slot reserves 'sqrt' for a future ablation."
            )

        self.k = k
        self.gamma_schedule = gamma_schedule
        self._t2_max = t2_max
        super().__init__(eps)

    @property
    def t2_max(self) -> float:
        """Upper bound on t_2 for training."""
        return self._t2_max

    def mu(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        t1: Tensor,        # [B, 1]
        t2: Tensor,        # [B, 1]
    ) -> Tensor:
        """Compute stacked interpolant mu(t_1, t_2).

        Returns (1-t_2)((1-t_1)x_0 + t_1 x_1) + t_2 x_*.
        """
        i_1 = (1.0 - t1) * x0 + t1 * x1          # [B, D]
        return (1.0 - t2) * i_1 + t2 * xstar      # [B, D]

    def dmu_dt1(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        t1: Tensor,        # [B, 1]
        t2: Tensor,        # [B, 1]
    ) -> Tensor:
        """Compute partial mu / partial t_1.

        Returns (1-t_2)(x_1 - x_0).
        """
        return (1.0 - t2) * (x1 - x0)             # [B, D]

    def dmu_dt2(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        t1: Tensor,        # [B, 1]
        t2: Tensor,        # [B, 1]
    ) -> Tensor:
        """Compute partial mu / partial t_2.

        Returns x_* - ((1-t_1) x_0 + t_1 x_1).
        """
        i_1 = (1.0 - t1) * x0 + t1 * x1          # [B, D]
        return xstar - i_1                         # [B, D]

    def gamma(self, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute noise amplitude (std) gamma(t_1, t_2).

        Linear-stiff schedule (independent of t_2):
            gamma = (1 - exp(-k t_1)) * (1 - exp(-k(1 - t_1)))

        Uses torch.expm1 for numerical stability at small t_1.
        Strictly positive on (0, 1) for k > 0.
        """
        # (-expm1(-x)) = 1 - exp(-x), stable near 0
        return (-torch.expm1(-self.k * t1)) * (-torch.expm1(-self.k * (1.0 - t1)))  # [B, 1]

    def dgamma_dt1(self, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute partial gamma / partial t_1.

        d/dt_1 [(1 - e1)(1 - e2)] = k e1 (1 - e2) - k e2 (1 - e1)
        where e1 = exp(-k t_1), e2 = exp(-k(1-t_1)).
        """
        e1 = torch.exp(-self.k * t1)               # [B, 1]
        e2 = torch.exp(-self.k * (1.0 - t1))       # [B, 1]
        return self.k * e1 * (1.0 - e2) - self.k * e2 * (1.0 - e1)  # [B, 1]

    def dgamma_dt2(self, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute partial gamma / partial t_2.

        Linear-stiff gamma is independent of t_2, so dgamma/dt_2 = 0 exactly.
        """
        return torch.zeros_like(t2)                # [B, 1]
