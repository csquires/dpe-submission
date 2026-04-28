"""Parameterized 2D inference curve in the time square.

The Curve2D class encapsulates a 1D inference curve tau in [0, 1] that maps to
points (t_1(tau), t_2(tau)) in the 2D time square [0, 1]^2. The default curve
is the LowArc, which forms a smooth arc below the diagonal, enabling V3's
line-integral inference through the interior of the (t_1, t_2) domain.
"""


class Curve2D:
    """Encapsulates a 1D parameterized curve tau -> (t_1(tau), t_2(tau)) in [0, 1]^2.

    Default curve is the LowArc: t_1(tau) = tau, t_2(tau) = h * tau * (1 - tau),
    where h = path_height. This curve starts at (0, 0), peaks at (0.5, h/4), and ends
    at (1, 0), tracing an arc below the diagonal.

    Constructor parameter:
      path_height: h >= 0. Controls the peak height of the arc. h = 0 collapses the
        curve to the bottom edge (t_2 = 0 for all tau), reducing to V2's stock 1D
        path along the t_1 axis. h = 1.0 default reaches peak t_2 = 0.25 at tau = 0.5.
        No validation; caller's responsibility.

    Method contract:
      - All methods take tau as a Python float in [0, 1] (typically from scipy
        solve_ivp's integration parameter).
      - All methods return a Python float.
      - Caller is responsible for casting tau/derivatives to tensors (with the right
        dtype/device) when used inside a model forward pass.

    Subclassing: future curves (e.g., diagonal tau -> (tau, tau)) can subclass
    Curve2D and override the four methods without changing the interface. No ABC at
    this stage since only LowArc ships in this PR.
    """

    def __init__(self, path_height: float = 1.0) -> None:
        """Initialize curve with given path_height.

        Args:
            path_height: float, default 1.0. Peak height multiplier for the LowArc.
                         Must be >= 0; not validated.
        """
        self.path_height = path_height

    def t1(self, tau: float) -> float:
        """Compute t_1(tau) = tau (identity map).

        Args:
            tau: float in [0, 1].

        Returns:
            float, the t_1 coordinate.
        """
        return float(tau)

    def t2(self, tau: float) -> float:
        """Compute t_2(tau) = h * tau * (1 - tau).

        Quadratic in tau; peaks at tau = 0.5 with value h/4.

        Args:
            tau: float in [0, 1].

        Returns:
            float, the t_2 coordinate.
        """
        return self.path_height * tau * (1.0 - tau)

    def dt1(self, tau: float) -> float:
        """Compute d t_1 / d tau = 1 (constant).

        Args:
            tau: float (unused for LowArc; kept for interface consistency with dt2
                 and future curve subclasses).

        Returns:
            float, constant 1.0.
        """
        return 1.0

    def dt2(self, tau: float) -> float:
        """Compute d t_2 / d tau = h * (1 - 2 tau).

        Linear in tau; zero at tau = 0.5 (peak of the arc).

        Args:
            tau: float in [0, 1].

        Returns:
            float, the derivative.
        """
        return self.path_height * (1.0 - 2.0 * tau)

    def peak_t2(self) -> float:
        """Return the maximum value t_2 reaches along the curve.

        Used by the estimator to assert that the path's t2_max covers the
        inference curve's t_2 range; without this check the network would be
        queried at untrained (t_1, t_2) values during predict_ldr.

        For LowArc t_2(tau) = h tau (1 - tau), the peak is at tau = 0.5
        with value h/4.

        Returns:
            float, the maximum t_2 attained on the curve.
        """
        return 0.25 * self.path_height
