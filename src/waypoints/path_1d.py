"""Continuous-time triangular interpolant path abstractions for score-based DRE.

Defines abstract interfaces (Path1D, CtsmPath1D, VfmPath1D) for parameterizing
continuous curves tau in [eps, 1-eps] through p0-to-p1 triangular mixtures.
Concrete implementations live in src/waypoints/triangular_continuous.py and beyond.
"""
from abc import ABC, abstractmethod
from torch import Tensor


class Path1D(ABC):
    """Root abstraction for continuous 1D-time triangular paths.

    Holds the shared eps boundary and establishes the semantic contract that paths
    parameterize a continuous curve from p0 through a p* mixture to p1.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        """Initialize path with boundary epsilon.

        Args:
            eps: boundary epsilon to exclude numerical singularities at tau=0 and tau=1.
                 Paths are valid in [eps, 1-eps]. Default 1e-3.
        """
        self.eps = eps


class CtsmPath1D(Path1D, ABC):
    """Contract for CTSM-style closed-form MSE regression on continuous triangular paths.

    Exposes the interface needed by score-based DRE methods that require sampling
    on the path and computing a per-sample regression target and weight.
    """

    @abstractmethod
    def sample_and_target(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
        epsilon: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample point on path and return closed-form regression target.

        Samples a point x_tau on the triangular interpolant path parameterized by tau,
        along with a closed-form regression target for MSE-based score/velocity training.
        Returns three tensors: the sampled point, the detached regression target, and
        per-sample weight factor lambda_t. The prediction paired with this target is:
            MSE(target - lambda_t * model(x_tau, tau)).

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p* (the midpoint mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps]. Caller MUST clamp.
            epsilon: [B, D] standard Gaussian noise ~ N(0, I). Named
                     `epsilon` (not `eps`) to avoid shadowing `self.eps` which
                     is the scalar tau-clamping boundary attribute from Path1D.

        Returns:
            x_tau: [B, D] sampled point on the path at time tau.
                          Tensor lives on same device as inputs.
            target: [B, 1] closed-form regression target, DETACHED from computation graph.
                           Caller assumes this is detached and will not backprop through it.
            lambda_t: [B, 1] per-sample weight factor (scale applied to model output),
                             DETACHED from computation graph.

        Notes:
            - Both target and lambda_t are detached; caller is responsible for gradient flow.
            - Caller is responsible for ensuring tau in [self.eps, 1-self.eps].
            - Returned tensors are on the same device as input tensors.
            - The paired prediction formula is MSE(target - lambda_t * model(x_tau, tau)).
        """


class VfmPath1D(Path1D, ABC):
    """Contract for VFM-style (velocity field matching) two-phase training on continuous triangular paths.

    Exposes the interface needed by velocity field matching methods that require
    sampling with stochasticity, drift computation, and derivatives.
    """

    @abstractmethod
    def sample(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
        z: Tensor,
    ) -> Tensor:
        """Sample point on path with stochasticity.

        Returns I_tau + gamma(tau) * z, where I_tau is the deterministic drift
        component (interpolant center) and gamma(tau) is the noise amplitude.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p* (the midpoint mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].
            z: [B, D] standard Gaussian noise ~ N(0, I).

        Returns:
            sample: [B, D] point on path. Tensor lives on same device as inputs.
        """

    @abstractmethod
    def drift(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """Compute deterministic drift (interpolant center I_tau).

        Returns the center of the interpolant path at time tau, without stochastic perturbation.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p* (the midpoint mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            drift: [B, D] deterministic interpolant center. Same device as inputs.
        """

    @abstractmethod
    def drift_deriv(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """Compute derivative of drift w.r.t. tau (dI_tau/dtau).

        Used for VFM velocity field training targets. Returns the time-derivative of the
        deterministic interpolant center I_tau.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p* (the midpoint mixture).
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            deriv: [B, D] time-derivative of drift. Same device as inputs.
        """

    @abstractmethod
    def gamma(self, tau: Tensor) -> Tensor:
        """Compute noise amplitude schedule gamma(tau).

        Returns the standard-deviation-like noise multiplier at time tau. Must be
        strictly positive on [self.eps, 1-self.eps]; zero or negative values are
        correctness bugs.

        Args:
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            gamma: [B, 1] noise amplitude. Strictly positive. Same device as input.

        Notes:
            - Must be strictly positive on [eps, 1-eps] for correctness.
            - For continuous closed-form paths this is always satisfied.
            - For piecewise paths this is a critical correctness gate.
        """

    @abstractmethod
    def dgamma_dtau(self, tau: Tensor) -> Tensor:
        """Compute derivative of gamma w.r.t. tau (dgamma/dtau).

        Time-derivative of the noise amplitude schedule. Used in VFM post-hoc time-score
        computations.

        Args:
            tau: [B, 1] time parameter in [self.eps, 1-self.eps].

        Returns:
            deriv: [B, 1] time-derivative of gamma. Same device as input.
        """
