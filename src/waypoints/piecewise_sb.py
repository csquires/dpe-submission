"""triangular piecewise Schroedinger-bridge paths for TriangularCTSM and TriangularVFM."""
import torch
from torch import Tensor
from src.waypoints.path_1d import CtsmPath1D, VfmPath1D
from src.waypoints.sb_bridge import sb_target


class PiecewiseSBCtsm1D(CtsmPath1D):
    """piecewise-SB interpolant x0 -> xstar -> x1 with hard switch at tau=vertex.

    leg 1 (tau in [0, vertex]):  local time t1 = tau / vertex,         x0 -> xstar.
    leg 2 (tau in [vertex, 1]):  local time t2 = (tau-vertex)/(1-vertex), xstar -> x1.
    each leg is a Gaussian SB; the canonical CTSM target uses uniform lambda_t=1
    and a torch.where(tau < vertex, ...) selector.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        vertex: float = 0.5,
        eps: float = 1e-3,
        inner_eps: float = 0.02,
    ) -> None:
        """inner_eps >= 0 protects each leg's vertex-side boundary so the per-leg
        variance never vanishes near tau=vertex. inner_eps=0 disables the guard."""
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
        """tau drawn from the vertex-free support; falls back to uniform [eps, 1-eps]
        when inner_eps == 0.

        when inner_eps > 0: pick leg by Bernoulli(vertex), then
            leg 1: t_local ~ U(eps, 1 - inner_eps);   tau = t_local vertex
            leg 2: t_local ~ U(inner_eps, 1 - eps);   tau = vertex + t_local (1 - vertex)
        forbidden band: (vertex - inner_eps vertex, vertex + inner_eps (1 - vertex)).
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
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
        epsilon: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """piecewise-SB sample and CTSM target with d/dtau chain-rule scaling.

        for tau < vertex use local t1 = tau / vertex on the x0 -> xstar leg;
        otherwise local t2 = (tau - vertex) / (1 - vertex) on xstar -> x1. each
        leg's target is scaled by 1/vertex or 1/(1-vertex) to convert d/dt_local
        back to d/dtau. lambda_t and x_tau are tau-parameterization invariant.
        epsilon is renamed (not `eps`) to avoid shadowing self.eps.
        """
        mask_leg1 = tau < self.vertex
        local_tau1 = tau / self.vertex
        local_tau2 = (tau - self.vertex) / (1 - self.vertex)
        local_tau1_clamped = torch.clamp(local_tau1, self.inner_eps, 1 - self.inner_eps)
        local_tau2_clamped = torch.clamp(local_tau2, self.inner_eps, 1 - self.inner_eps)

        x_tau1, target_local1, lambda_t_local1 = sb_target(
            x0=x0, x1=xstar, sigma=self.sigma, tau=local_tau1_clamped, epsilon=epsilon,
        )
        x_tau2, target_local2, lambda_t_local2 = sb_target(
            x0=xstar, x1=x1, sigma=self.sigma, tau=local_tau2_clamped, epsilon=epsilon,
        )

        target_scaled1 = target_local1 / self.vertex
        target_scaled2 = target_local2 / (1 - self.vertex)

        x_tau = torch.where(mask_leg1, x_tau1, x_tau2)
        target = torch.where(mask_leg1, target_scaled1, target_scaled2)
        lambda_t = torch.where(mask_leg1, lambda_t_local1, lambda_t_local2)
        return x_tau, target.detach(), lambda_t.detach()


class PiecewiseSBVfm1D(VfmPath1D):
    """piecewise-SB VFM interpolant with hard-floored gamma.

    each leg has SB stochasticity gamma_raw = sigma sqrt(t_leg (1 - t_leg)). gamma
    is then clamped from below by gamma_min, which makes the noise variance flat
    in an O((gamma_min/sigma)^2 min(vertex, 1-vertex)) band around tau=vertex. the
    floored path is still Gaussian and Bayes-consistent, so the LDR remains
    asymptotically correct; bias inside the clamp window is O(gamma_min^2 |b||eta|).
    in that window dgamma/dtau is forced to zero (returned as such by dgamma_dtau).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        vertex: float = 0.5,
        gamma_min: float = 5e-2,
        eps: float = 1e-3,
        inner_eps: float = 0.0,
    ) -> None:
        """eps must be >= 1e-3 for boundary regularity; inner_eps > 0 excludes a
        width-inner_eps band around tau=vertex from sample_tau."""
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
        """tau drawn from the vertex-free support (mirrors PiecewiseSBCtsm1D.sample_tau)."""
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
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
        z: Tensor,
    ) -> Tensor:
        """mu_tau + gamma(tau) z with the per-leg linear drift."""
        t1 = tau / self.vertex
        drift_leg1 = (1 - t1) * x0 + t1 * xstar
        t2 = (tau - self.vertex) / (1 - self.vertex)
        drift_leg2 = (1 - t2) * xstar + t2 * x1
        drift_t = torch.where(tau < self.vertex, drift_leg1, drift_leg2)
        return drift_t + self.gamma(tau) * z

    def drift(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """per-leg linear interpolant mu_tau."""
        t1 = tau / self.vertex
        drift_leg1 = (1 - t1) * x0 + t1 * xstar
        t2 = (tau - self.vertex) / (1 - self.vertex)
        drift_leg2 = (1 - t2) * xstar + t2 * x1
        return torch.where(tau < self.vertex, drift_leg1, drift_leg2)

    def drift_deriv(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """dmu/dtau: (xstar - x0)/vertex on leg 1, (x1 - xstar)/(1 - vertex) on leg 2."""
        delta_leg1 = (xstar - x0) / self.vertex
        delta_leg2 = (x1 - xstar) / (1 - self.vertex)
        return torch.where(tau < self.vertex, delta_leg1, delta_leg2)

    def gamma(self, tau: Tensor) -> Tensor:
        """gamma(tau) = max(per-leg SB gamma, gamma_min); strictly positive on [eps, 1-eps]."""
        t1 = tau / self.vertex
        t2 = (tau - self.vertex) / (1 - self.vertex)
        gamma_leg1 = self.sigma * torch.sqrt(t1 * (1 - t1))
        gamma_leg2 = self.sigma * torch.sqrt(t2 * (1 - t2))
        gamma_raw = torch.where(tau < self.vertex, gamma_leg1, gamma_leg2)
        return torch.clamp_min(gamma_raw, self.gamma_min)

    def dgamma_dtau(self, tau: Tensor) -> Tensor:
        """dgamma/dtau, forced to zero inside the gamma_min clamp window.

        the clamp test on gamma_raw is run before the SB formula to keep the
        analytical 1/sqrt(t(1-t)) expression away from the t in {0, 1} boundary.
        """
        t1 = tau / self.vertex
        t2 = (tau - self.vertex) / (1 - self.vertex)
        gamma_leg1_raw = self.sigma * torch.sqrt(t1 * (1 - t1))
        gamma_leg2_raw = self.sigma * torch.sqrt(t2 * (1 - t2))
        mask = tau < self.vertex
        gamma_raw = torch.where(mask, gamma_leg1_raw, gamma_leg2_raw)
        clamped = gamma_raw < self.gamma_min

        d_leg1 = (self.sigma / self.vertex) * (1 - 2 * t1) / (2 * torch.sqrt(t1 * (1 - t1)))
        d_leg2 = (self.sigma / (1 - self.vertex)) * (1 - 2 * t2) / (2 * torch.sqrt(t2 * (1 - t2)))
        dgamma_raw = torch.where(mask, d_leg1, d_leg2)
        return torch.where(clamped, torch.zeros_like(dgamma_raw), dgamma_raw)
