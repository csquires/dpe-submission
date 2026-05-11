"""Triangular piecewise-Schrodinger-bridge paths (V1) for score-based DRE.

Implements PiecewiseSBCtsm1D (CtsmPath1D subclass) for V1 of TriangularCTSM,
and PiecewiseSBVfm1D (VfmPath1D subclass) for V1 of TriangularVFM. Both use
hard-switch per-leg targets (no crossfade) and canonical time-score with
uniform lambda=1 (CTSM) or hard-floored gamma schedule (VFM).
"""
import torch
from torch import Tensor
from src.waypoints.path_1d import CtsmPath1D, VfmPath1D
from src.waypoints.sb_bridge import sb_bridge_target


class PiecewiseSBCtsm1D(CtsmPath1D):
    """Triangular piecewise-SB interpolant with closed-form canonical CTSM target.

    Parameterizes a continuous path from x0 to x1 via intermediate anchor xstar.
    The path consists of two legs: leg 1 (tau in [0, vertex]) interpolates from
    x0 to xstar via local time t1 = tau / vertex; leg 2 (tau in [vertex, 1])
    interpolates from xstar to x1 via local time t2 = (tau - vertex) / (1 - vertex).
    Each leg is Gaussian with per-leg SB stochasticity (linear variance schedule
    per local time).

    The closed-form target is the canonical conditional time-score T^* computed
    per-leg with uniform weight lambda_t = 1. A hard boolean switch (no crossfade)
    selects between legs based on tau < vertex.

    No crossfade_width parameter; hard switch via torch.where(mask, leg1, leg2).
    Vertex generalization: any vertex in (0, 1) is supported.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        vertex: float = 0.5,
        eps: float = 1e-3,
        inner_eps: float = 0.02,
    ) -> None:
        """Initialize triangular piecewise-SB CTSM path.

        Args:
            sigma: scalar float > 0, noise scale parameter. Default 1.0.
            vertex: scalar float in (0, 1), location of switch between legs. Default 0.5.
            eps: scalar float > 0, lower/upper clamp for tau boundary. Default 1e-3.
                 Must satisfy eps < min(vertex, 1 - vertex).
                 Passed to super().__init__(eps).
            inner_eps: scalar float >= 0, local-time floor on each leg's INNER
                 (vertex-side) boundary. Default 0.02. With this guard, the
                 per-leg variance never vanishes near tau=vertex, so the
                 closed-form CTSM target stays bounded. Empirical A/B sweep
                 showed monotone improvement vs inner_eps=0 across mild and
                 sharp Gaussian KLs and pendulum (the original 82%-divergence
                 regime); plateau reached at 0.02 with no further gains up to
                 0.20. Set to 0.0 to recover the legacy unguarded behavior.

        Raises:
            ValueError: if sigma <= 0.
            ValueError: if vertex not in (0, 1).
            ValueError: if eps <= 0.
            ValueError: if eps >= min(vertex, 1 - vertex).
            ValueError: if inner_eps < 0 or inner_eps + eps >= 1.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if not (0 < vertex < 1):
            raise ValueError(f"vertex must be in (0, 1), got {vertex}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if eps >= min(vertex, 1 - vertex):
            raise ValueError(
                f"eps must be < min(vertex, 1-vertex) = {min(vertex, 1 - vertex)}, got {eps}"
            )
        if inner_eps < 0:
            raise ValueError(f"inner_eps must be >= 0, got {inner_eps}")
        if inner_eps + eps >= 1.0:
            raise ValueError(
                f"inner_eps ({inner_eps}) + eps ({eps}) must be < 1; "
                f"otherwise the per-leg t_local interval is empty"
            )

        self.sigma = sigma
        self.vertex = vertex
        self.inner_eps = inner_eps
        super().__init__(eps)

    def sample_tau(
        self,
        batch_size: int,
        eps: float,
        device,
    ):
        """sample tau ~ U over the *vertex-free* support when inner_eps > 0.

        Falls back to uniform on [eps, 1-eps] when inner_eps == 0 (current
        behavior). When inner_eps > 0, draw a leg index proportional to leg
        width, then a per-leg t_local sample with two-sided protection: the
        OUTER boundary (tau -> 0 in leg 1, tau -> 1 in leg 2) is guarded by
        `eps`; the INNER boundary (tau -> vertex on either leg) is guarded
        by `inner_eps`.

        Per-leg t_local distributions:
            leg 1:  t_local ~ U(eps, 1 - inner_eps)
            leg 2:  t_local ~ U(inner_eps, 1 - eps)

        Mapping:
            tau_leg1 = t_local * vertex
            tau_leg2 = vertex + t_local * (1 - vertex)

        The forbidden band around vertex is therefore
        (vertex - inner_eps * vertex, vertex + inner_eps * (1 - vertex)).

        Returns: [B, 1] tensor of tau values in [eps, 1 - eps].
        """
        import torch

        if self.inner_eps <= 0.0:
            return (
                torch.rand(batch_size, 1, device=device)
                * (1.0 - 2.0 * eps)
                + eps
            )
        leg1_mask = torch.rand(batch_size, 1, device=device) < self.vertex
        # leg 1: t_local in [eps, 1 - inner_eps] -> tau in [eps*v, (1-inner)*v]
        t_local_leg1 = (
            torch.rand(batch_size, 1, device=device)
            * (1.0 - self.inner_eps - eps)
            + eps
        )
        # leg 2: t_local in [inner_eps, 1 - eps] -> tau in [v + inner*(1-v), v + (1-eps)*(1-v)]
        t_local_leg2 = (
            torch.rand(batch_size, 1, device=device)
            * (1.0 - eps - self.inner_eps)
            + self.inner_eps
        )
        tau_leg1 = t_local_leg1 * self.vertex
        tau_leg2 = self.vertex + t_local_leg2 * (1.0 - self.vertex)
        return torch.where(leg1_mask, tau_leg1, tau_leg2)

    def sample_and_target(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
        epsilon: Tensor    # [B, D]; named epsilon (not eps) to avoid shadowing self.eps
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample point on piecewise-SB path and return closed-form CTSM regression target.

        Per-leg local times: t1 = tau / vertex (leg 1), t2 = (tau - vertex) / (1 - vertex) (leg 2).
        Delegates per-leg SB mathematics to sb_bridge_target helper with chain-rule scaling
        applied to targets (1/vertex for leg1, 1/(1-vertex) for leg2) to convert from local
        tau derivatives back to global tau derivatives. lambda_t and x_tau are NOT scaled
        (lambda_t is variance-based, tau-parameterization invariant; x_tau is a path value).

        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p* (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps]. Caller must pre-clamp.
            epsilon: [B, D] standard Gaussian noise ~ N(0, I).

        Returns:
            x_tau: [B, D] sampled point on path. Not detached.
            target: [B, 1] closed-form regression target, detached.
            lambda_t: [B, 1] per-sample weight factor, detached.
        """
        # leg membership: mask = tau < vertex (boolean, [B, 1])
        mask_leg1 = tau < self.vertex                         # [B, 1]

        # compute local tau on each leg
        local_tau1 = tau / self.vertex                        # [B, 1]
        local_tau2 = (tau - self.vertex) / (1 - self.vertex)  # [B, 1]

        # clamp local tau to [inner_eps, 1 - inner_eps] for vertex-band protection
        local_tau1_clamped = torch.clamp(local_tau1, self.inner_eps, 1 - self.inner_eps)  # [B, 1]
        local_tau2_clamped = torch.clamp(local_tau2, self.inner_eps, 1 - self.inner_eps)  # [B, 1]

        # delegate to sb_bridge_target for each leg
        x_tau1, target_local1, lambda_t_local1 = sb_bridge_target(
            x_start=x0,
            x_end=xstar,
            sigma=self.sigma,
            tau=local_tau1_clamped,
            epsilon=epsilon
        )  # all [B, D] or [B, 1]

        x_tau2, target_local2, lambda_t_local2 = sb_bridge_target(
            x_start=xstar,
            x_end=x1,
            sigma=self.sigma,
            tau=local_tau2_clamped,
            epsilon=epsilon
        )  # all [B, D] or [B, 1]

        # apply chain-rule scaling to target only (not lambda_t or x_tau)
        # chain rule: d/dtau = (1/vertex) * d/dt_local for leg 1
        target_scaled1 = target_local1 * (1.0 / self.vertex)   # [B, 1]
        # chain rule: d/dtau = (1/(1-vertex)) * d/dt_local for leg 2
        target_scaled2 = target_local2 * (1.0 / (1 - self.vertex))  # [B, 1]

        # hard switch to select per-sample outputs (no crossfade)
        x_tau = torch.where(mask_leg1, x_tau1, x_tau2)        # [B, D]
        target = torch.where(mask_leg1, target_scaled1, target_scaled2)  # [B, 1]
        lambda_t = torch.where(mask_leg1, lambda_t_local1, lambda_t_local2)  # [B, 1]

        return x_tau, target.detach(), lambda_t.detach()


class PiecewiseSBVfm1D(VfmPath1D):
    """Triangular piecewise-SB interpolant with hard-floored VFM gamma schedule.

    Parameterizes a continuous path from x0 to x1 via intermediate anchor xstar.
    The path consists of two legs (tau in [0, vertex] and [vertex, 1]), each with
    per-leg SB stochasticity: gamma_raw_leg = sigma * sqrt(t_leg * (1 - t_leg)).

    Hard gamma floor: gamma_t = max(gamma_raw, gamma_min) via torch.clamp_min.
    This modifies the path family: in a tau-band of width O((gamma_min/sigma)^2 * vertex)
    around tau = vertex on each leg, the noise variance is clamped to gamma_min^2
    (locally flat, not SB). The floored path is still Gaussian and Bayes-consistent;
    the LDR estimate recovers log(p_0 / p_1) asymptotically. The bias from the
    clamp window is O(gamma_min^2 * |b||eta|), with clamp-window tau-width
    O((gamma_min/sigma)^2 * min(vertex, 1-vertex)) (depends on the narrower leg).

    Vertex generalization: any vertex in (0, 1) is supported (Q4 unlock).
    Default gamma_min = 5e-2 chosen empirically to balance b-network convergence
    and LDR accuracy on Gaussian toys (vertex sweep and gamma_min sweep at
    acceptance gate time).

    Note on dgamma_dtau: the clamp window causes dgamma/dtau to be zero where
    gamma_raw < gamma_min. This is detached in the estimator (see
    triangular_vfm.py:212), so the b-target regresses Delta alone in the clamp
    region (valid probability-flow velocity for locally flat noise). Future
    differentiable callers would silently lose gradients on the clamped region.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        vertex: float = 0.5,
        gamma_min: float = 5e-2,
        eps: float = 1e-3,
        inner_eps: float = 0.0,
    ) -> None:
        """Initialize triangular piecewise-SB VFM path.

        Args:
            sigma: scalar float > 0, noise scale parameter. Default 1.0.
            vertex: scalar float in (0, 1), location of switch between legs. Default 0.5.
            gamma_min: scalar float > 0, hard floor on gamma schedule. Default 5e-2.
            eps: scalar float >= 1e-3, lower/upper clamp for tau boundary. Default 1e-3.
                 Must satisfy eps >= 1e-3 (boundary regularity floor) and
                 eps < min(vertex, 1 - vertex). Passed to super().__init__(eps).
            inner_eps: scalar float >= 0, local-time floor on each leg's INNER
                 (vertex-side) boundary. Default 0.0 means no guard. If > 0,
                 sample_tau() draws t_local with two-sided protection so the
                 sampled tau distribution excludes a width-`inner_eps` band
                 around the vertex. mirrors the PiecewiseSBCtsm1D fix.

        Raises:
            ValueError: if sigma <= 0.
            ValueError: if vertex not in (0, 1).
            ValueError: if gamma_min <= 0.
            ValueError: if eps < 1e-3.
            ValueError: if eps >= min(vertex, 1 - vertex).
            ValueError: if inner_eps < 0 or inner_eps + eps >= 1.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if not (0 < vertex < 1):
            raise ValueError(f"vertex must be in (0, 1), got {vertex}")
        if gamma_min <= 0:
            raise ValueError(f"gamma_min must be > 0, got {gamma_min}")
        if eps < 1e-3:
            raise ValueError(f"eps must be >= 1e-3 (boundary regularity floor), got {eps}")
        if eps >= min(vertex, 1 - vertex):
            raise ValueError(
                f"eps must be < min(vertex, 1-vertex) = {min(vertex, 1 - vertex)}, got {eps}"
            )
        if inner_eps < 0:
            raise ValueError(f"inner_eps must be >= 0, got {inner_eps}")
        if inner_eps + eps >= 1.0:
            raise ValueError(
                f"inner_eps ({inner_eps}) + eps ({eps}) must be < 1; "
                f"otherwise per-leg t_local interval is empty"
            )

        self.sigma = sigma
        self.vertex = vertex
        self.gamma_min = gamma_min
        self.inner_eps = inner_eps
        super().__init__(eps)

    def sample_tau(
        self,
        batch_size: int,
        eps: float,
        device,
    ):
        """sample tau ~ U over the *vertex-free* support when inner_eps > 0.

        Mirrors PiecewiseSBCtsm1D.sample_tau exactly: per-leg two-sided
        protection. Returns [B, 1] tensor.
        """
        import torch

        if self.inner_eps <= 0.0:
            return (
                torch.rand(batch_size, 1, device=device)
                * (1.0 - 2.0 * eps)
                + eps
            )
        leg1_mask = torch.rand(batch_size, 1, device=device) < self.vertex
        t_local_leg1 = (
            torch.rand(batch_size, 1, device=device)
            * (1.0 - self.inner_eps - eps)
            + eps
        )
        t_local_leg2 = (
            torch.rand(batch_size, 1, device=device)
            * (1.0 - eps - self.inner_eps)
            + self.inner_eps
        )
        tau_leg1 = t_local_leg1 * self.vertex
        tau_leg2 = self.vertex + t_local_leg2 * (1.0 - self.vertex)
        return torch.where(leg1_mask, tau_leg1, tau_leg2)

    def sample(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
        z: Tensor,         # [B, D]
    ) -> Tensor:
        """Sample point on piecewise-SB path with stochasticity.

        Returns mu_tau + gamma_t * z, where mu_tau is the deterministic drift
        (per-leg linear interpolant) and gamma_t is the hard-floored noise amplitude.

        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p* (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].
            z: [B, D] standard Gaussian noise ~ N(0, I).

        Returns:
            sample: [B, D] point on path. Not detached. Same device as inputs.
        """
        # per-leg drift (linear interpolants)
        t1_local = tau / self.vertex                                # [B, 1]
        drift_leg1 = (1 - t1_local) * x0 + t1_local * xstar        # [B, D]

        t2_local = (tau - self.vertex) / (1 - self.vertex)          # [B, 1]
        drift_leg2 = (1 - t2_local) * xstar + t2_local * x1         # [B, D]

        # hard switch for drift
        mask = tau < self.vertex                                    # [B, 1]
        drift_t = torch.where(mask, drift_leg1, drift_leg2)         # [B, D]

        # gamma (hard floored)
        gamma_t = self.gamma(tau)                                   # [B, 1]

        # sample
        return drift_t + gamma_t * z                                # [B, D]

    def drift(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
    ) -> Tensor:
        """Compute deterministic drift mu_tau (per-leg linear interpolant).

        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p* (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            drift: [B, D] deterministic interpolant center. Same device as inputs.
        """
        t1_local = tau / self.vertex                                # [B, 1]
        drift_leg1 = (1 - t1_local) * x0 + t1_local * xstar        # [B, D]

        t2_local = (tau - self.vertex) / (1 - self.vertex)          # [B, 1]
        drift_leg2 = (1 - t2_local) * xstar + t2_local * x1         # [B, D]

        mask = tau < self.vertex                                    # [B, 1]
        return torch.where(mask, drift_leg1, drift_leg2)            # [B, D]

    def drift_deriv(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
    ) -> Tensor:
        """Compute derivative of drift w.r.t. tau (dmu/dtau = Delta).

        Per-leg Delta: (xstar - x0) / vertex for leg 1, (x1 - xstar) / (1 - vertex) for leg 2.
        Hard switch via mask = tau < vertex.

        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p* (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            deriv: [B, D] time-derivative of drift. Same device as inputs.
        """
        Delta_leg1 = (xstar - x0) / self.vertex                     # [B, D]
        Delta_leg2 = (x1 - xstar) / (1 - self.vertex)               # [B, D]

        mask = tau < self.vertex                                    # [B, 1]
        return torch.where(mask, Delta_leg1, Delta_leg2)            # [B, D]

    def gamma(self, tau: Tensor) -> Tensor:
        """Compute hard-floored noise amplitude gamma(tau).

        Per-leg SB stochasticity: gamma_raw_leg = sigma * sqrt(t_leg * (1 - t_leg)) where
        t_leg is the local time [0, 1] on that leg. Hard floor: gamma_t = max(gamma_raw, gamma_min).

        Strictly positive on [self.eps, 1-self.eps] by construction (gamma_min > 0 and
        per-leg SB > 0 in the unclamped region).

        Args:
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            gamma: [B, 1] noise amplitude. Strictly positive. Same device as input.
        """
        # per-leg SB gamma
        t1_local = tau / self.vertex                                # [B, 1]
        gamma_leg1 = self.sigma * torch.sqrt(t1_local * (1 - t1_local))  # [B, 1]

        t2_local = (tau - self.vertex) / (1 - self.vertex)          # [B, 1]
        gamma_leg2 = self.sigma * torch.sqrt(t2_local * (1 - t2_local))  # [B, 1]

        # hard switch
        mask = tau < self.vertex                                    # [B, 1]
        gamma_raw = torch.where(mask, gamma_leg1, gamma_leg2)       # [B, 1]

        # hard floor
        gamma_t = torch.clamp_min(gamma_raw, self.gamma_min)        # [B, 1]

        return gamma_t

    def dgamma_dtau(self, tau: Tensor) -> Tensor:
        """Compute derivative of gamma w.r.t. tau (dgamma/dtau).

        Per-leg SB derivative:
          dgamma_raw/dtau = (sigma / 2) * (1 - 2 * t_leg) / sqrt(t_leg * (1 - t_leg)) / leg_width

        Inside the clamp window (where gamma_raw < gamma_min, strict inequality), return zero.
        Outside, return the analytical per-leg SB derivative. The clamp condition is evaluated
        before computing the derivative to avoid division-by-zero on the boundary.

        WARNING: Currently this function returns zero inside the clamp window. If a future caller
        requires gradients of dgamma_dtau w.r.t. tau, those gradients will silently be lost in
        the clamped region. The current callers (triangular_vfm.py) call .detach() on the
        returned tensor, so no gradient loss occurs; this warning is a future-proofing note.

        Args:
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            deriv: [B, 1] time-derivative of gamma. Zero inside clamp window, analytical
                   derivative outside. Same device as input.
        """
        # per-leg SB raw gamma (before floor)
        t1_local = tau / self.vertex                                       # [B, 1]
        gamma_leg1_raw = self.sigma * torch.sqrt(t1_local * (1 - t1_local))  # [B, 1]

        t2_local = (tau - self.vertex) / (1 - self.vertex)                 # [B, 1]
        gamma_leg2_raw = self.sigma * torch.sqrt(t2_local * (1 - t2_local))  # [B, 1]

        # hard switch to select per-leg gamma_raw
        mask = tau < self.vertex                                           # [B, 1]
        gamma_raw = torch.where(mask, gamma_leg1_raw, gamma_leg2_raw)      # [B, 1]

        # identify clamped region: strict inequality gamma_raw < gamma_min
        clamped = gamma_raw < self.gamma_min                               # [B, 1] boolean

        # per-leg analytical derivatives (SB formula)
        # leg 1: dgamma_raw/dtau = (sigma / vertex) * (1 - 2 * t1_local) / (2 * sqrt(t1_local * (1 - t1_local)))
        t1_product = t1_local * (1 - t1_local)                             # [B, 1]
        dgamma_leg1_raw_dtau = (
            (self.sigma / self.vertex)
            * (1 - 2 * t1_local)
            / (2 * torch.sqrt(t1_product))
        )                                                                   # [B, 1]

        # leg 2: dgamma_raw/dtau = (sigma / (1 - vertex)) * (1 - 2 * t2_local) / (2 * sqrt(t2_local * (1 - t2_local)))
        t2_product = t2_local * (1 - t2_local)                             # [B, 1]
        dgamma_leg2_raw_dtau = (
            (self.sigma / (1 - self.vertex))
            * (1 - 2 * t2_local)
            / (2 * torch.sqrt(t2_product))
        )                                                                   # [B, 1]

        # hard switch to select per-leg derivative
        dgamma_raw = torch.where(mask, dgamma_leg1_raw_dtau, dgamma_leg2_raw_dtau)  # [B, 1]

        # clamp window mask: inside (clamped=True) return zero, outside (clamped=False) return dgamma_raw
        dgamma_t = torch.where(clamped, torch.zeros_like(dgamma_raw), dgamma_raw)  # [B, 1]

        return dgamma_t
