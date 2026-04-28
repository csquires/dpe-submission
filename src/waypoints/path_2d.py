"""Continuous-time triangular interpolant path abstractions for 2D-time score-based DRE.

Defines abstract interfaces (Path2D, CtsmPath2D) for parameterizing continuous
surfaces over (t_1, t_2) in [eps, 1-eps]^2 through p0-to-p1 triangular mixtures
with anchor p_*. Concrete implementations live in src/waypoints/triangular_continuous_2d.py.

Mirrors the 1D analog at src/waypoints/path_1d.py: a Path2D root carries the eps
boundary, and CtsmPath2D adds the closed-form-target contract used by V3 of
TriangularCTSM. Mirrors the 1D analog at src/waypoints/path_1d.py: CtsmPath2D and VfmPath2D
ship as root ABCs for CTSM and VFM training paradigms, respectively.
Concrete implementations live in src/waypoints/triangular_continuous_2d.py.
"""
from abc import ABC, abstractmethod
from torch import Tensor


class Path2D(ABC):
    """Root abstraction for continuous 2D-time triangular paths.

    Holds the shared eps boundary and establishes the semantic contract that paths
    parameterize a continuous surface over (t_1, t_2) in [eps, 1-eps]^2 from p0 (at
    (0, 0)) through a p_* mixture (toward t_2 = 1) to p_1 (at (1, 0)).
    """

    def __init__(self, eps: float = 1e-3) -> None:
        """Initialize path with boundary epsilon.

        Args:
            eps: boundary epsilon to exclude numerical singularities at the corners
                 of [0, 1]^2. Paths are valid in [eps, 1-eps]^2. Default 1e-3.
        """
        self.eps = eps


class CtsmPath2D(Path2D, ABC):
    """Contract for CTSM-style closed-form MSE regression on continuous 2D-time triangular paths.

    Exposes the interface needed by score-based DRE methods that require sampling
    on the path and computing a 2-vector closed-form regression target and weight,
    one component per time coordinate t_i.
    """

    @abstractmethod
    def sample_and_target(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        t1: Tensor,
        t2: Tensor,
        epsilon: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample point on the 2D-time path and return closed-form 2-vector regression target.

        Samples a point x on the triangular interpolant surface at (t_1, t_2),
        along with a closed-form 2-vector regression target (one component per t_i)
        for MSE-based score training. The prediction paired with this target is:
            MSE(target - lambda_t * model(x, t_1, t_2))
        where the model outputs shape [B, 2] matching the target.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p_* (the anchor).
            t1: [B, 1] first time coordinate. Caller MUST clamp to [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate. Caller MUST clamp to [self.eps, 1-self.eps]
                or to a further-restricted range read from the concrete path object.
            epsilon: [B, D] standard Gaussian noise N(0, I). Named `epsilon` (not `eps`)
                     to avoid shadowing self.eps.

        Returns:
            x: [B, D] sampled point on the path. NOT detached.
            target: [B, 2] closed-form 2-vector regression target. DETACHED.
            lambda_t: [B, 2] per-component, per-sample weight factor. DETACHED.

        Notes:
            - target and lambda_t are detached; caller is responsible for gradient flow.
            - x is not detached; gradient flows through model(x, t_1, t_2).
            - Component i corresponds to the closed-form target paired with
              the partial derivative \\partial_{t_i} \\log \\rho.
            - Caller is responsible for tau bounds; this method does not validate.
        """


class VfmPath2D(Path2D, ABC):
    """Contract for VFM-style velocity field matching on continuous 2D-time triangular paths.

    Exposes the interface needed by velocity field matching methods that require
    sampling with stochasticity, 2D drift computation and time-partials, noise
    amplitude schedule, and its time-partials. Uses a path parameterization:
        x = \\mu(t_1, t_2) + \\gamma(t_1, t_2) z
    where z ~ N(0, I_d). VFM training uses \\partial_{t_i} \\mu and \\partial_{t_i} \\gamma
    to define velocity targets b_i^* = E[\\partial_{t_i} \\mu + (\\partial_{t_i} \\gamma) z | x].
    VFM inference divides by \\gamma, so concrete subclasses must guarantee \\gamma > 0
    strictly on [eps, 1-eps] x [eps, t2_max].
    """

    @abstractmethod
    def mu(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        t1: Tensor,
        t2: Tensor,
    ) -> Tensor:
        """Compute drift mean \\mu(t_1, t_2).

        Returns the deterministic center of the 2D-time triangular interpolant
        at (t_1, t_2), without stochastic perturbation.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p_* (the anchor).
            t1: [B, 1] first time coordinate in [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate in [self.eps, t2_max].

        Returns:
            mu: [B, D] drift mean at (t_1, t_2). Same device as inputs.
        """

    @abstractmethod
    def dmu_dt1(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        t1: Tensor,
        t2: Tensor,
    ) -> Tensor:
        """Compute time-derivative of drift w.r.t. t_1: \\partial \\mu / \\partial t_1.

        Used for VFM velocity field training targets in the t_1 direction.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p_* (the anchor).
            t1: [B, 1] first time coordinate in [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate in [self.eps, t2_max].

        Returns:
            deriv: [B, D] partial \\mu / \\partial t_1 at (t_1, t_2). Same device as inputs.
        """

    @abstractmethod
    def dmu_dt2(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        t1: Tensor,
        t2: Tensor,
    ) -> Tensor:
        """Compute time-derivative of drift w.r.t. t_2: \\partial \\mu / \\partial t_2.

        Used for VFM velocity field training targets in the t_2 direction.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p_* (the anchor).
            t1: [B, 1] first time coordinate in [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate in [self.eps, t2_max].

        Returns:
            deriv: [B, D] partial \\mu / \\partial t_2 at (t_1, t_2). Same device as inputs.
        """

    @abstractmethod
    def gamma(self, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute noise amplitude schedule \\gamma(t_1, t_2).

        Returns the standard-deviation-like noise multiplier at time (t_1, t_2).
        Must be strictly positive on [self.eps, 1-eps] x [self.eps, t2_max];
        zero or negative values are correctness bugs for VFM inference.

        Args:
            t1: [B, 1] first time coordinate in [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate in [self.eps, t2_max].

        Returns:
            gamma: [B, 1] noise amplitude. Strictly positive. Same device as inputs.

        Notes:
            - Must be strictly positive on the valid domain for correctness.
            - For continuous closed-form paths this is always satisfied.
            - For piecewise paths this is a critical correctness gate.
        """

    @abstractmethod
    def dgamma_dt1(self, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute time-derivative of gamma w.r.t. t_1: \\partial \\gamma / \\partial t_1.

        Used in VFM post-hoc time-score computations.

        Args:
            t1: [B, 1] first time coordinate in [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate in [self.eps, t2_max].

        Returns:
            deriv: [B, 1] partial \\gamma / \\partial t_1 at (t_1, t_2). Same device as inputs.
        """

    @abstractmethod
    def dgamma_dt2(self, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute time-derivative of gamma w.r.t. t_2: \\partial \\gamma / \\partial t_2.

        Used in VFM post-hoc time-score computations. May be identically zero
        for schedules with no t_2 dependence.

        Args:
            t1: [B, 1] first time coordinate in [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate in [self.eps, t2_max].

        Returns:
            deriv: [B, 1] partial \\gamma / \\partial t_2 at (t_1, t_2). Same device as inputs.
        """

    def sample(
        self,
        x0: Tensor,
        x1: Tensor,
        xstar: Tensor,
        t1: Tensor,
        t2: Tensor,
        z: Tensor,
    ) -> Tensor:
        """Sample point on 2D-time path with stochasticity.

        Returns \\mu(t_1, t_2) + \\gamma(t_1, t_2) z, where \\mu is the deterministic
        drift and \\gamma is the noise amplitude. NOT detached; gradient flows through
        the drift and noise terms.

        Args:
            x0: [B, D] bootstrap-sampled endpoint from p0.
            x1: [B, D] bootstrap-sampled endpoint from p1.
            xstar: [B, D] bootstrap-sampled point from p_* (the anchor).
            t1: [B, 1] first time coordinate in [self.eps, 1-self.eps].
            t2: [B, 1] second time coordinate in [self.eps, t2_max].
            z: [B, D] standard Gaussian noise ~ N(0, I).

        Returns:
            sample: [B, D] point on path. Not detached. Same device as inputs.
        """
        return self.mu(x0, x1, xstar, t1, t2) + self.gamma(t1, t2) * z  # [B, D]

    @property
    @abstractmethod
    def t2_max(self) -> float:
        """Upper bound on t_2 for training.

        The 2D-time domain is [self.eps, 1-self.eps] x [self.eps, t2_max].
        Read by the estimator when sampling t_2 uniformly or scheduling.

        Returns:
            float: Upper bound on t_2. Must be > self.eps and finitely representable.
        """
