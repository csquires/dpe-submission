"""Triangular continuous-time Schrodinger-bridge path using barycentric interpolation.

Implements BarycentricCtsm1D (CtsmPath1D subclass) for V2 of TriangularCTSM.
The closed-form target is the canonical conditional time-score T^*.
"""
import torch
from torch import Tensor
from src.waypoints.path_1d import CtsmPath1D, VfmPath1D


def _barycentric_weights(
    tau: Tensor,         # [B, 1]
    vertex: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """compute the C^1 triangular barycentric weights and their tau-derivatives.

    asymmetric piecewise-quadratic bell with peak at tau=vertex and h(0)=h(1)=0:
        h(tau; v) = (tau/v)(2 - tau/v)             for tau <= v
                  = ((1-tau)/(1-v))(2 - (1-tau)/(1-v))   for tau >  v
    h is C^1 (h'(v) = 0 from both sides) and reduces to 4*tau*(1-tau) at v=0.5.
    weights are alpha = (1-tau)(1-h), beta = tau(1-h), w_* = h, summing to 1
    by construction; their derivatives sum to 0. shared by BarycentricCtsm1D
    and BarycentricVfm1D so any bug-fix or schedule swap propagates to both.

    note: the noise-variance schedule (g_t = tau*(1-tau) for ctsm; gamma(tau)
    for vfm) is NOT shifted by vertex; only the anchor-weight bell is.

    Args:
        tau: time parameter, broadcastable shape (typically [B, 1]).
        vertex: float in (0, 1), location of the bell peak. Default 0.5
            preserves the legacy symmetric behavior.

    Returns:
        (alpha, beta, w_*, dot{alpha}, dot{beta}, dot{w_*}), each same shape as tau.
    """
    if vertex == 0.5:
        # legacy fast-path; byte-identical to pre-asymmetric-bell behavior.
        h = 4 * tau * (1 - tau)
        h_prime = 4 * (1 - 2 * tau)
    else:
        # general piecewise-quadratic bell with peak at v.
        v = vertex
        u_left = tau / v
        u_right = (1 - tau) / (1 - v)
        h_left = u_left * (2.0 - u_left)
        h_right = u_right * (2.0 - u_right)
        h = torch.where(tau <= v, h_left, h_right)
        # left:  dh/dtau = (2/v)(1 - tau/v)
        # right: dh/dtau = -(2/(1-v))(1 - (1-tau)/(1-v))
        h_prime_left = (2.0 / v) * (1.0 - u_left)
        h_prime_right = -(2.0 / (1.0 - v)) * (1.0 - u_right)
        h_prime = torch.where(tau <= v, h_prime_left, h_prime_right)

    alpha = (1 - tau) * (1 - h)
    beta = tau * (1 - h)
    w_star = h

    alpha_prime = -(1 - h) - (1 - tau) * h_prime
    beta_prime = (1 - h) - tau * h_prime
    w_star_prime = h_prime

    return alpha, beta, w_star, alpha_prime, beta_prime, w_star_prime


class BarycentricCtsm1D(CtsmPath1D):
    """Triangular barycentric interpolant with Gaussian noise and canonical CTSM target.

    Parameterizes a continuous path from x0 to x1 via intermediate anchor xstar,
    using barycentric weights with a triangular profile h(tau) = 4*tau*(1-tau).
    Noise variance vanishes at tau=0 and tau=1, suitable for score-matching training.

    The closed-form target is the conditional time-score
        T^* = (\dot V / (2 V)) (|epsilon|^2 - d) + (Delta . epsilon) / sqrt{V}
    with uniform per-sample weight lambda_t = 1.
    """

    def __init__(self, sigma: float = 1.0, vertex: float = 0.5, eps: float = 1e-3):
        """Initialize triangular CTSM path.

        Args:
            sigma: scalar float > 0, noise scale parameter. Default 1.0.
            vertex: scalar float in (0, 1), location of peak w_star influence. Default 0.5.
                    Must be in (0, 1). Default 0.5 preserves the legacy
                    symmetric bell.
            eps: scalar float, lower/upper clamp for tau boundary. Default 1e-3.
                 Passed to super().__init__(eps).

        Raises:
            ValueError: if vertex not in (0, 1).
            ValueError: if sigma ≤ 0.
        """
        if not (0.0 < vertex < 1.0):
            raise ValueError(f"vertex must be in (0, 1), got {vertex}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self.sigma = sigma
        self.vertex = vertex
        super().__init__(eps)

    def sample_and_target(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
        epsilon: Tensor    # [B, D]; named epsilon (not eps) to avoid shadowing self.eps
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample point on triangular path and compute closed-form CTSM regression target.

        Generates a point x_tau on the barycentric interpolant path at time tau,
        perturbed by Gaussian noise with variance that vanishes at endpoints.
        Returns the point and a closed-form regression target for MSE score-matching.

        The path uses three anchor points (x0, x1, xstar) weighted by:
            alpha_t = (1 - tau) * (1 - h_t)    (weight for x0)
            beta_t = tau * (1 - h_t)           (weight for x1)
            w_star_t = h_t                     (weight for xstar)
        where h_t = 4 * tau * (1 - tau) is the triangular bell.

        The noise amplitude std_t = sigma * sqrt(tau * (1 - tau)) vanishes at endpoints.

        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p_star (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].
            epsilon: [B, D] standard Gaussian noise ~ N(0, I).

        Returns:
            x_tau: [B, D] sampled point on path. Not detached.
            target: [B, 1] closed-form regression target, detached.
            lambda_t: [B, 1] per-sample weight factor, detached.
        """
        # barycentric weights and their derivatives (shared helper)
        alpha_t, beta_t, w_star_t, alpha_prime_t, beta_prime_t, w_star_prime_t = (
            _barycentric_weights(tau, self.vertex)
        )

        # noise variance and its derivative
        g_t = tau * (1 - tau)                       # [B, 1]
        dg_dtau_t = 1 - 2 * tau                     # [B, 1]
        std_t = self.sigma * torch.sqrt(g_t)        # [B, 1]

        # path mean and drift direction
        mu_tau = alpha_t * x0 + beta_t * x1 + w_star_t * xstar  # [B, D]
        Delta = alpha_prime_t * x0 + beta_prime_t * x1 + w_star_prime_t * xstar  # [B, D]

        # noisy sample
        x_tau = mu_tau + std_t * epsilon            # [B, D]

        # closed-form target computation
        epsilon_sq = (epsilon ** 2).sum(dim=-1, keepdim=True)   # [B, 1]
        delta_dot_epsilon = (Delta * epsilon).sum(dim=-1, keepdim=True)  # [B, 1]
        dim = epsilon.shape[-1]

        # variance schedule
        var_t = self.sigma ** 2 * g_t               # [B, 1]
        d_var_t = self.sigma ** 2 * dg_dtau_t       # [B, 1]

        # data-bounded normalization (mirrors plain CTSM):
        #   target   = T_raw / temp,  lambda_t = var_t / temp
        # where temp = sqrt(2 ||x_1 - x_0||^2 + 1e-8). prior version used
        # `target = T_raw / var_t, lambda_t = 1`, which made the network
        # chase infinite targets at tau -> 0,1 (heavy-tailed gradients,
        # L^2-unbounded loss). bayes-optimal predictor pred = T_raw / var_t
        # is unchanged at the optimum; only the loss conditioning improves.
        delta_endpoint_sq = ((x1 - x0) ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
        temp = torch.sqrt(2 * delta_endpoint_sq + 1e-8)                 # [B, 1]

        target = (d_var_t * (epsilon_sq - dim) / 2.0
                  + std_t * delta_dot_epsilon) / temp
        lambda_t = var_t / temp

        return x_tau, target.detach(), lambda_t.detach()


class BarycentricVfm1D(VfmPath1D):
    """Triangular barycentric interpolant with VFM gamma schedule.

    Parameterizes a continuous path from x_0 to x_1 via intermediate anchor x_*,
    using barycentric weights with a triangular bell h(tau) = 4 tau (1-tau).
    Noise amplitude gamma(tau) follows the existing VFM schedule and is strictly
    positive on (0, 1).

    The drift is mu_tau = alpha(tau) x_0 + beta(tau) x_1 + w_*(tau) x_*.
    The drift derivative Delta_tau = dot{alpha} x_0 + dot{beta} x_1 + dot{w_*} x_*
    vanishes at tau = 0.5 (all weight derivatives are zero at the vertex). This
    is by design and benign: at tau = 0.5 the path is locally stationary at x_*.
    """

    def __init__(self, k: float = 20.0, vertex: float = 0.5, eps: float = 1e-3):
        """Initialize BarycentricVfm1D.

        Args:
            k: noise schedule parameter (>0). Default 20.0 matches stock VFM HPO range.
            vertex: vertex location of barycentric bell, in (0, 1). Default
                0.5 preserves the legacy symmetric bell.
            eps: tau clamp boundary; must be >= 1e-3 (boundary regularity floor).

        Raises:
            ValueError: if vertex not in (0, 1).
            ValueError: if k <= 0.
            ValueError: if eps < 1e-3.
        """
        if not (0.0 < vertex < 1.0):
            raise ValueError(f"vertex must be in (0, 1), got {vertex}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if eps < 1e-3:
            raise ValueError(f"eps must be >= 1e-3 (boundary regularity floor), got {eps}")

        self.k = k
        self.vertex = vertex
        super().__init__(eps)

    def sample(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
        z: Tensor,         # [B, D]
    ) -> Tensor:
        """Sample point on VFM triangular path with stochasticity.

        Returns mu_tau + gamma(tau) * z, where mu_tau is the deterministic drift
        (barycentric interpolant) and gamma(tau) is the VFM noise amplitude schedule.
        NOT detached.

        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p* (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps]. Caller must pre-clamp.
            z: [B, D] standard Gaussian noise ~ N(0, I).

        Returns:
            sample: [B, D] point on path. Not detached. Same device as inputs.
        """
        alpha_t, beta_t, w_star_t, *_ = _barycentric_weights(tau, self.vertex)
        mu_tau = alpha_t * x0 + beta_t * x1 + w_star_t * xstar  # [B, D]
        gamma_t = self.gamma(tau)                               # [B, 1]
        return mu_tau + gamma_t * z                             # [B, D]

    def drift(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
    ) -> Tensor:
        """Compute deterministic drift mu_tau = alpha*x0 + beta*x1 + w_*  *xstar.

        The drift is the barycentric interpolant center without stochastic perturbation.

        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p* (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            drift: [B, D] deterministic interpolant center. Same device as inputs.
        """
        alpha_t, beta_t, w_star_t, *_ = _barycentric_weights(tau, self.vertex)
        return alpha_t * x0 + beta_t * x1 + w_star_t * xstar  # [B, D]

    def drift_deriv(
        self,
        x0: Tensor,        # [B, D]
        x1: Tensor,        # [B, D]
        xstar: Tensor,     # [B, D]
        tau: Tensor,       # [B, 1]
    ) -> Tensor:
        """Compute derivative of drift w.r.t. tau: Delta_tau = dot{alpha}*x0 + dot{beta}*x1 + dot{w_*}*xstar.

        Time-derivative of the deterministic interpolant center. Used for VFM velocity
        field training targets.

        Note:
            At tau = 0.5 (the vertex), Delta_{0.5} = 0 identically. This is by design
            (all weight derivatives vanish at the triangular peak) and benign: the path
            is locally stationary at x_* at the vertex.
        Args:
            x0: [B, D] endpoint from p0.
            x1: [B, D] endpoint from p1.
            xstar: [B, D] point from p* (intermediate mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            deriv: [B, D] time-derivative of drift. Same device as inputs.
        """
        *_, alpha_prime_t, beta_prime_t, w_star_prime_t = _barycentric_weights(tau, self.vertex)
        return alpha_prime_t * x0 + beta_prime_t * x1 + w_star_prime_t * xstar  # [B, D]

    def gamma(self, tau: Tensor) -> Tensor:
        """Compute noise amplitude schedule gamma(tau).

        Returns the VFM noise amplitude at time tau. Uses torch.expm1 for numerical
        stability of (1 - exp(-k*tau)) at small tau.

        Formula: gamma(tau) = (-expm1(-k*tau)) * (-expm1(-k*(1-tau)))
                           = (1 - exp(-k*tau)) * (1 - exp(-k*(1-tau)))

        Strictly positive on [self.eps, 1-self.eps] by construction.

        Args:
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            gamma: [B, 1] noise amplitude. Strictly positive. Same device as input.
        """
        return (-torch.expm1(-self.k * tau)) * (-torch.expm1(-self.k * (1 - tau)))  # [B, 1]

    def dgamma_dtau(self, tau: Tensor) -> Tensor:
        """Compute derivative of gamma w.r.t. tau: dgamma/dtau.

        Time-derivative of the VFM noise amplitude schedule. Used in VFM post-hoc
        time-score computations.

        Formula: dgamma/dtau = k*exp(-k*tau)*(1-exp(-k*(1-tau)))
                             - k*exp(-k*(1-tau))*(1-exp(-k*tau))

        Args:
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            deriv: [B, 1] time-derivative of gamma. Same device as input.
        """
        exp_kt = torch.exp(-self.k * tau)              # [B, 1]
        exp_k1t = torch.exp(-self.k * (1 - tau))       # [B, 1]

        return self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)  # [B, 1]
