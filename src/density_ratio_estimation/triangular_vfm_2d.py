"""TriangularVFM2D: V3-VFM 2D-time stacked-interpolant density ratio estimator.

Mirrors TriangularVFM (1D) but trains two velocity heads (b_1, b_2) and one
denoiser (eta) sequentially on a 2D-time stacked interpolant path. Inference
integrates the time-score along a Curve2D from tau=eps to 1-eps.
"""
from typing import Optional, Literal
import warnings
import itertools

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.waypoints.path_2d import VfmPath2D
from src.waypoints.triangular_continuous_2d import Stacked2DVfm
from src.waypoints.curve_2d import Curve2D
from src.models.time_score_matching.velocity_network_2d import MLP2D


class TriangularVFM2D(DensityRatioEstimator):
    """V3-VFM 2D-time triangular VFM density ratio estimator.

    Trains two velocity heads (b_1, b_2) and one denoiser (eta) sequentially
    on three distributions p_0, p_1, p_*. Inference integrates the time-score
    along self.curve from tau=eps to 1-eps.

    Contract: fit(samples_p0, samples_p1, samples_pstar) with three [N, D]
    tensors; predict_ldr(xs) returns log(p_0/p_1) as [n_samples] CPU tensor.
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[VfmPath2D] = None,
        curve: Optional[Curve2D] = None,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1.3e-3,
        eps: float = 1e-3,
        device: Optional[str] = None,
        integration_steps: int = 3000,
        integration_type: Literal['1', '2', '3'] = '2',
        antithetic: bool = True,
        verbose: bool = False,
        log_every: int = 100,
    ) -> None:
        """Initialize TriangularVFM2D.

        Args:
            input_dim: Spatial dimension D.
            path: VfmPath2D instance. If None, default Stacked2DVfm(eps=eps).
            curve: Curve2D instance. If None, default Curve2D(path_height=1.0).
            hidden_dim: MLP2D hidden width.
            n_epochs: Training epochs per phase.
            batch_size: Minibatch size.
            lr: Adam learning rate.
            eps: Boundary margin for tau / t_1 sampling. Must be >= 1e-3.
            device: Device string. Auto-resolves if None.
            integration_steps: Number of tau quadrature points.
            integration_type: '1' mean, '2' trapz, '3' Simpson.
            antithetic: Toggle antithetic variance reduction in b-phase.
            verbose: Toggle epoch-level logging.
            log_every: Epochs between log prints.
        """
        # blocking validation: must be FIRST
        if eps < 1e-3:
            raise ValueError(
                f"eps must be >= 1e-3 for boundary regularity of b*eta/gamma; "
                f"got eps={eps}"
            )

        super().__init__(input_dim)

        # store hyperparameters
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.integration_steps = integration_steps
        self.integration_type = integration_type
        self.antithetic = antithetic
        self.verbose = verbose
        self.log_every = log_every

        # resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # resolve path / curve defaults
        self.path = path if path is not None else Stacked2DVfm(
            k=20.0, gamma_schedule="linear-stiff", t2_max=0.3, eps=eps
        )
        self.curve = curve if curve is not None else Curve2D(path_height=1.0)

        # coverage assertion: curve t_2 range must lie within trained t_2 range
        peak = float(self.curve.peak_t2())
        t2_max = float(self.path.t2_max)
        assert peak <= t2_max + 1e-9, (
            f"curve peak_t2 {peak} exceeds path.t2_max {t2_max}"
        )

        # network placeholders
        self.net_b1 = None
        self.net_b2 = None
        self.net_eta = None

    def init_model(self) -> None:
        """Instantiate three independent MLP2D networks on self.device."""
        self.net_b1 = MLP2D(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)
        self.net_b2 = MLP2D(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)
        self.net_eta = MLP2D(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """Train b_1, b_2, eta networks sequentially on three distributions.

        Args:
            samples_p0: [N0, D] samples from p_0.
            samples_p1: [N1, D] samples from p_1.
            samples_pstar: [Nstar, D] samples from p_*.

        Procedure:
            phase 1: joint b_1, b_2 optimizer (eta frozen) — velocity matching loss.
            phase 2: eta optimizer (b_1, b_2 frozen) — denoising loss.
        """
        n_star = samples_pstar.shape[0]

        if n_star < 1:
            raise ValueError(f"samples_pstar must have at least 1 row; got n_star={n_star}")

        if n_star < self.batch_size // 4:
            warnings.warn(
                f"n_star={n_star} is small relative to batch_size={self.batch_size}; "
                f"pstar bootstrap will sample with high replication"
            )

        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        self.init_model()

        if self.verbose:
            print("[TriangularVFM2D] Starting Sequential Training (3 distributions)")

        self._train_b_phase(samples_p0, samples_p1, samples_pstar)
        self._train_eta_phase(samples_p0, samples_p1, samples_pstar)

        # post-training cleanup
        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()

    def _train_b_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """Joint training of b_1 and b_2 with eta frozen.

        Per-direction losses computed separately; sum is back-propagated.
        Antithetic variance reduction is applied if self.antithetic=True.
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b1.train()
        self.net_b2.train()
        self.net_eta.eval()
        optimizer_b = optim.Adam(
            itertools.chain(self.net_b1.parameters(), self.net_b2.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        t2_max = float(self.path.t2_max)

        for epoch in range(self.n_epochs):
            # bootstrap minibatches
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # time sampling
            t1 = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*self.eps) + self.eps  # [B, 1]
            t2 = torch.rand(self.batch_size, 1, device=self.device) * (t2_max - self.eps) + self.eps  # [B, 1]

            # noise
            z = torch.randn_like(x0)  # [B, D]

            # path quantities (detached)
            mu = self.path.mu(x0, x1, xstar, t1, t2).detach()  # [B, D]
            dmu_dt1 = self.path.dmu_dt1(x0, x1, xstar, t1, t2).detach()  # [B, D]
            dmu_dt2 = self.path.dmu_dt2(x0, x1, xstar, t1, t2).detach()  # [B, D]
            gamma_t = self.path.gamma(t1, t2).detach()  # [B, 1]
            dgamma_dt1 = self.path.dgamma_dt1(t1, t2).detach()  # [B, 1]
            dgamma_dt2 = self.path.dgamma_dt2(t1, t2).detach()  # [B, 1]

            if self.antithetic:
                # antithetic variance reduction: evaluate at +z and -z
                x_t_plus = mu + gamma_t * z  # [B, D]
                x_t_minus = mu - gamma_t * z  # [B, D]

                b1_plus = self.net_b1(t1, t2, x_t_plus)  # [B, D]
                b2_plus = self.net_b2(t1, t2, x_t_plus)  # [B, D]
                b1_minus = self.net_b1(t1, t2, x_t_minus)  # [B, D]
                b2_minus = self.net_b2(t1, t2, x_t_minus)  # [B, D]

                target_1_plus = dmu_dt1 + dgamma_dt1 * z  # [B, D]
                target_2_plus = dmu_dt2 + dgamma_dt2 * z  # [B, D]
                target_1_minus = dmu_dt1 - dgamma_dt1 * z  # [B, D]
                target_2_minus = dmu_dt2 - dgamma_dt2 * z  # [B, D]

                # half-norm-minus-dot per direction, averaged over +/- pair
                loss_b1 = (
                    0.25 * (b1_plus ** 2).sum(dim=-1)
                    - 0.5 * (target_1_plus * b1_plus).sum(dim=-1)
                    + 0.25 * (b1_minus ** 2).sum(dim=-1)
                    - 0.5 * (target_1_minus * b1_minus).sum(dim=-1)
                ).mean()
                loss_b2 = (
                    0.25 * (b2_plus ** 2).sum(dim=-1)
                    - 0.5 * (target_2_plus * b2_plus).sum(dim=-1)
                    + 0.25 * (b2_minus ** 2).sum(dim=-1)
                    - 0.5 * (target_2_minus * b2_minus).sum(dim=-1)
                ).mean()
            else:
                x_t = mu + gamma_t * z  # [B, D]
                b1_pred = self.net_b1(t1, t2, x_t)  # [B, D]
                b2_pred = self.net_b2(t1, t2, x_t)  # [B, D]

                target_1 = dmu_dt1 + dgamma_dt1 * z  # [B, D]
                target_2 = dmu_dt2 + dgamma_dt2 * z  # [B, D]

                loss_b1 = (
                    0.5 * (b1_pred ** 2).sum(dim=-1)
                    - (target_1 * b1_pred).sum(dim=-1)
                ).mean()
                loss_b2 = (
                    0.5 * (b2_pred ** 2).sum(dim=-1)
                    - (target_2 * b2_pred).sum(dim=-1)
                ).mean()

            loss = loss_b1 + loss_b2

            optimizer_b.zero_grad()
            loss.backward()
            optimizer_b.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                # log per-direction losses (NOT summed)
                print(f"  [Epoch {epoch+1}] loss_b1={loss_b1.item():.4f} loss_b2={loss_b2.item():.4f}")

    def _train_eta_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """Train eta with b_1 and b_2 frozen.

        Mirrors V2-VFM eta-phase (denoising loss).
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.train()
        optimizer_eta = optim.Adam(
            self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )

        t2_max = float(self.path.t2_max)

        for epoch in range(self.n_epochs):
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            t1 = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*self.eps) + self.eps  # [B, 1]
            t2 = torch.rand(self.batch_size, 1, device=self.device) * (t2_max - self.eps) + self.eps  # [B, 1]

            z = torch.randn_like(x0)  # [B, D]
            x_t = self.path.sample(x0, x1, xstar, t1, t2, z).detach()  # [B, D]

            eta_pred = self.net_eta(t1, t2, x_t)  # [B, D]

            # denoising loss: half-norm minus dot with z
            loss_eta = (
                0.5 * (eta_pred ** 2).sum(dim=-1) - (z * eta_pred).sum(dim=-1)
            ).mean()

            optimizer_eta.zero_grad()
            loss_eta.backward()
            optimizer_eta.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_eta={loss_eta.item():.4f}")

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Estimate \\log p_0(x) / p_1(x) via time-score line integral on self.curve.

        This estimator computes TWO divergences (one per velocity head) per
        integration step. The 2x cost relative to V2-VFM is intentional: the
        2D-time decomposition splits the time-score into two directional
        components, each requiring its own div-of-velocity. Do NOT collapse
        the two jacrev calls into one — they have different inputs and outputs.

        Args:
            xs: [N, D] test points (CPU or device); will be moved to self.device.

        Returns:
            [N] log density ratios, CPU float32.

        Raises:
            RuntimeError: if any of self.net_b1, self.net_b2, self.net_eta is None.
        """
        if self.net_b1 is None or self.net_b2 is None or self.net_eta is None:
            raise RuntimeError(
                "TriangularVFM2D model is not trained. Call fit() before predict_ldr()."
            )

        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)  # [n_samples, D]
        n_samples = samples.shape[0]

        # tau grid (odd for Simpson)
        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        tau_vals = torch.linspace(self.eps, 1.0 - self.eps, steps=n_points, device=self.device)  # [n_points]

        # step 5a: pack curve outputs into [n_points, 4] BEFORE vmap
        curve = self.curve
        tau_list = tau_vals.tolist()  # n_points python floats
        t_data = torch.tensor(
            [[curve.t1(tau), curve.t2(tau), curve.dt1(tau), curve.dt2(tau)] for tau in tau_list],
            device=self.device,
            dtype=samples.dtype,
        )  # [n_points, 4]

        # chunked inference: vmap over leading dim of t_data
        chunk_size = max(1, 100000 // n_samples)
        compute_vmapped = torch.vmap(
            self._compute_time_score_single,
            in_dims=(0, None),
            out_dims=0,
        )

        time_score_chunks = []
        for i in range(0, n_points, chunk_size):
            t_chunk = t_data[i:i + chunk_size]  # [chunk_len, 4]
            chunk_scores = compute_vmapped(t_chunk, samples).detach()  # [chunk_len, n_samples]
            time_score_chunks.append(chunk_scores)

        time_scores = torch.cat(time_score_chunks, dim=0)  # [n_points, n_samples]

        if self.integration_type == '2':
            out = -torch.trapz(time_scores, tau_vals, dim=0).cpu()  # [n_samples]
        elif self.integration_type == '3':
            # simpson's rule (mirror V2-VFM)
            t_np = tau_vals.cpu().numpy()
            h = (t_np[-1] - t_np[0]) / (n_points - 1)
            integrand = time_scores.cpu().numpy()  # [n_points, n_samples]
            integral = integrand[0] + integrand[-1]
            for i in range(1, n_points - 1):
                if i % 2 == 0:
                    integral += 2 * integrand[i]
                else:
                    integral += 4 * integrand[i]
            integral *= h / 3
            out = -torch.from_numpy(integral)  # [n_samples]
        elif self.integration_type == '1':
            out = -time_scores.mean(dim=0).cpu()  # [n_samples]

        return out  # [n_samples], CPU float32

    def _compute_time_score_single(
        self, t_tau: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute time-score d log rho / d tau at a single tau.

        Mirrors V2-VFM `_compute_time_score_single`; the only structural
        difference is the packed `[4]`-vector input form (4 scalars instead
        of 1).

        Args:
            t_tau: [4] packed (t_1, t_2, dt_1/dtau, dt_2/dtau) as 0-d slices
                   produced by the outer vmap over [n_points, 4].
            x: [n_samples, D] test points (broadcast — not vmapped).

        Returns:
            [n_samples] time scores at the current tau.
        """
        # unpack 0-d tensors
        t1_s = t_tau[0]
        t2_s = t_tau[1]
        dt1_s = t_tau[2]
        dt2_s = t_tau[3]

        n_samples = x.shape[0]

        # 0-d tensors must be view(1, 1)'d before expand to [n_samples, 1]
        t1_batch = t1_s.view(1, 1).expand(n_samples, 1)  # [n_samples, 1]
        t2_batch = t2_s.view(1, 1).expand(n_samples, 1)  # [n_samples, 1]

        # gamma at the scalar (t_1, t_2): use [1, 1]-shaped tensors for path interface
        gamma_t = self.path.gamma(t1_s.view(1, 1), t2_s.view(1, 1)).squeeze()  # 0-d

        # network forwards (full batch)
        b1_pred = self.net_b1(t1_batch, t2_batch, x)  # [n_samples, D]
        b2_pred = self.net_b2(t1_batch, t2_batch, x)  # [n_samples, D]
        eta_pred = self.net_eta(t1_batch, t2_batch, x)  # [n_samples, D]

        # divergence via vmap(jacrev) per head — TWO separate calls, do not collapse
        t1_one = t1_s.view(1, 1)
        t2_one = t2_s.view(1, 1)

        def b1_single(x_single):
            return self.net_b1(t1_one, t2_one, x_single.unsqueeze(0)).squeeze(0)

        def b2_single(x_single):
            return self.net_b2(t1_one, t2_one, x_single.unsqueeze(0)).squeeze(0)

        def jac_trace_b1(x_single):
            return torch.trace(torch.func.jacrev(b1_single)(x_single))

        def jac_trace_b2(x_single):
            return torch.trace(torch.func.jacrev(b2_single)(x_single))

        div_b1 = torch.vmap(jac_trace_b1)(x)  # [n_samples]
        div_b2 = torch.vmap(jac_trace_b2)(x)  # [n_samples]

        # dot products
        b1_dot_eta = (b1_pred * eta_pred).sum(dim=-1)  # [n_samples]
        b2_dot_eta = (b2_pred * eta_pred).sum(dim=-1)  # [n_samples]

        # directional time-score components
        s_1 = -div_b1 + b1_dot_eta / gamma_t  # [n_samples]
        s_2 = -div_b2 + b2_dot_eta / gamma_t  # [n_samples]

        # combine via curve derivatives (chain rule)
        time_score = s_1 * dt1_s + s_2 * dt2_s  # [n_samples]
        return time_score


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5.0

    # gaussian pair with controlled KL
    gp = create_two_gaussians_kl(dim=DIM, k=KL_DIVERGENCE, beta=0.5)
    mu0, Sigma0 = gp["mu0"], gp["Sigma0"]
    mu1, Sigma1 = gp["mu1"], gp["Sigma1"]

    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    # midpoint anchor p_*
    mu_star = (mu0 + mu1) / 2.0
    Sigma_star = (Sigma0 + Sigma1) / 2.0
    pstar = MultivariateNormal(mu_star, covariance_matrix=Sigma_star)

    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar = pstar.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # smoke test with reduced epochs
    estimator = TriangularVFM2D(input_dim=DIM, verbose=True, n_epochs=200)
    estimator.fit(samples_p0, samples_p1, samples_pstar)

    est_ldrs = estimator.predict_ldr(samples_test)
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))

    print(f"MAE: {mae:.6f}")
