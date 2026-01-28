import torch


class TriangularWaypointBuilder1D:
    """
    Build a 1D triangular path p0 -> p* -> p1 using a single parameter t in [0, 1].
    """
    def __init__(self, midpoint_oversample: int = 0, gamma_power: float = 1.0) -> None:
        self.midpoint_oversample = max(0, int(midpoint_oversample))
        self.gamma_power = float(gamma_power)

    def _make_t(self, num_waypoints: int, device: torch.device) -> torch.Tensor:
        if self.midpoint_oversample <= 0 or num_waypoints <= 2:
            return torch.linspace(0, 1, num_waypoints, device=device)
        a = 1.0 + self.midpoint_oversample
        t_inner = torch.distributions.Beta(a, a).sample((num_waypoints - 2,)).to(device)
        t = torch.cat([torch.tensor([0.0, 1.0], device=device), t_inner], dim=0)
        return torch.sort(t).values

    def build_waypoints(
        self,
        samples_p0: torch.Tensor,     # [b0, dim]
        samples_p1: torch.Tensor,     # [b1, dim]
        samples_pstar: torch.Tensor,  # [bstar, dim]
        num_waypoints: int,
    ) -> torch.Tensor:
        t = self._make_t(num_waypoints, samples_p0.device).view(-1, 1)
        alpha = torch.clamp(1 - 2 * t, min=0.0)
        beta = torch.clamp(2 * t - 1, min=0.0)
        gamma = (1 - 2 * torch.abs(t - 0.5)).clamp(min=0.0)
        if self.gamma_power != 1.0:
            gamma = gamma ** self.gamma_power

        b0, dim = samples_p0.shape
        b1, _ = samples_p1.shape
        bstar, _ = samples_pstar.shape
        b = max(b0, b1, bstar)

        waypoint_samples = torch.zeros((num_waypoints, b, dim), device=samples_p0.device)
        for i in range(num_waypoints):
            new_samples_p0 = samples_p0[torch.randint(0, b0, (b,))]
            new_samples_p1 = samples_p1[torch.randint(0, b1, (b,))]
            new_samples_pstar = samples_pstar[torch.randint(0, bstar, (b,))]
            waypoint_samples[i] = (
                alpha[i] * new_samples_p0
                + beta[i] * new_samples_p1
                + gamma[i] * new_samples_pstar
            )
        return waypoint_samples  # [w, b, dim]
