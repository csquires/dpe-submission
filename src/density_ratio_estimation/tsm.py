"""Time Score Matching (TSM) density ratio estimator."""

from typing import Optional
import warnings

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._trainer import train_score_flow
from src.density_ratio_estimation._ema import sample_time_and_iw
from src.density_ratio_estimation._losses import hyvarinen_time_score_loss
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class TSM(DensityRatioEstimator):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        reweight: bool = False,
        eps: float = 1e-5,
        device: Optional[str] = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        n_hidden_layers: int = 3,
        activation: str = "silu",
        integration_steps: int = 200,
    ) -> None:
        """
        Time Score Matching density ratio estimator.

        Estimates log density ratio via a time-interpolated score-matching objective.
        Learns a time-dependent score network s_phi(x, tau) that is integrated over
        tau to predict log(p0(x) / p1(x)).

        Procedure:
          1. Constructor: validate activation, set hyperparameters, auto-detect device.
          2. fit(samples_p0, samples_p1): initialize model, delegate training loop to
             train_score_flow with hyvarinen_time_score_loss.
          3. predict_ldr(xs): integrate score network output via torch.trapezoid.

        Args:
            input_dim: Dimension of input space.
            hidden_dim: Width of hidden layers in score network. Default 256.
            n_epochs: Number of gradient steps (training loop iteration count).
                      Internally treated as n_steps for train_score_flow.
                      Kept for backward compatibility with HPO scripts. Default 1000.
            batch_size: Mini-batch size for each gradient step. Default 512.
            lr: Learning rate for Adam optimizer. Default 1e-3.
            reweight: If True, scale loss by Hyvärinen weighting lambda(tau).
                      Default False.
            eps: Time-domain margin; tau sampled from [eps, 1-eps].
                 Default 1e-5.
            device: Device string ("cuda", "cpu") or None for auto-detect.
                    If None, use cuda if available else cpu. Default None.
            rtol: ODE solver tolerance (deprecated; kept for HPO compat).
                  Emits DeprecationWarning if rtol != 1e-6. Default 1e-6.
            atol: ODE solver tolerance (deprecated; kept for HPO compat).
                  Emits DeprecationWarning if atol != 1e-6. Default 1e-6.
            n_hidden_layers: Number of hidden layers in score network. Default 3.
            activation: Activation function name in {"elu", "gelu", "silu"}.
                        Default "silu".
            integration_steps: Number of quadrature points for tau grid in predict_ldr.
                               Default 200.

        Raises:
            ValueError: If activation not in {"elu", "gelu", "silu"}.
        """
        super().__init__(input_dim)
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )

        # emit DeprecationWarning for rtol/atol (no longer used in torch.trapezoid path)
        if rtol != 1e-6:
            warnings.warn(
                "rtol is deprecated; torch.trapezoid does not use ODE tolerances. "
                "Parameter is accepted but ignored. Use integration_steps to control "
                "quadrature precision.",
                DeprecationWarning,
                stacklevel=2,
            )
        if atol != 1e-6:
            warnings.warn(
                "atol is deprecated; torch.trapezoid does not use ODE tolerances. "
                "Parameter is accepted but ignored. Use integration_steps to control "
                "quadrature precision.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reweight = reweight
        self.eps = eps
        self.rtol = rtol
        self.atol = atol
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.integration_steps = integration_steps

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
        """
        Instantiate TimeScoreNetwork1D and Adam optimizer.

        Procedure:
          1. Create TimeScoreNetwork1D with stored hyperparameters.
          2. Move model to self.device.
          3. Create Adam optimizer with lr=self.lr, betas=(0.9, 0.999), eps=1e-8.
        """
        self.model = TimeScoreNetwork1D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """
        Train the score network via train_score_flow + hyvarinen_time_score_loss.

        Procedure:
          1. Call init_model() to instantiate model and optimizer.
          2. Build time_sampler: lambda with signature (batch_size, eps, device) ->
             (tau [B,1], iw [B,1]), delegates to sample_time_and_iw("uniform", ...).
          3. Build loss_kwargs: {"reweight": self.reweight, "eps": self.eps}.
          4. Call train_score_flow with:
             - model: self.model
             - samples_p0, samples_p1: input samples
             - samples_pstar: None (TSM uses 2-source loss, no intermediate anchor)
             - loss_fn: hyvarinen_time_score_loss
             - optim: self.optimizer
             - n_steps: self.n_epochs (renamed for train_score_flow)
             - batch_size: self.batch_size
             - time_sampler: the lambda above
             - scheduler: None (no lr scheduling for TSM)
             - ema: None (no EMA for TSM)
             - grad_clip_norm: None (no grad clipping for TSM)
             - eps: self.eps
             - loss_kwargs: as built above

        Args:
            samples_p0: Source distribution samples, shape [N0, D].
            samples_p1: Target distribution samples, shape [N1, D].

        Raises:
            RuntimeError: If model instantiation fails or loss computation raises.
        """
        self.init_model()

        # time_sampler: uniform tau in [eps, 1-eps] with unit importance weights
        time_sampler = lambda B, eps_val, dev: sample_time_and_iw(
            "uniform", B, eps_val, dev
        )

        # loss_kwargs for hyvarinen_time_score_loss
        loss_kwargs = {"reweight": self.reweight, "eps": self.eps}

        # delegate training loop to unified trainer
        train_score_flow(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=hyvarinen_time_score_loss,
            optim=self.optimizer,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=None,
            ema=None,
            grad_clip_norm=None,
            eps=self.eps,
            loss_kwargs=loss_kwargs,
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Integrate learned score to predict log density ratio.

        Procedure:
          1. Validate model is not None; raise RuntimeError if untrained.
          2. Set model to eval mode.
          3. Move input to device and cast to float.
          4. Build uniform tau grid: ts = torch.linspace(eps, 1.0, integration_steps,
             device=device).
          5. For each tau in ts: evaluate model(xs, tau_broadcast) to get score,
             negate to form integrand -score(x, tau).
          6. Stack evaluations: vals [T, N_test].
          7. Integrate via torch.trapezoid over tau dimension; dt = (1.0 - eps) / (integration_steps - 1).
          8. Return result as CPU tensor.

        Tensor Shapes:
          - xs: [N_test, D] input.
          - ts: [T] where T = integration_steps.
          - tau_broadcast: [N_test, 1] (repeated for each tau in ts).
          - model(xs, tau_broadcast): [N_test] score output.
          - vals: [T, N_test] stacked scores.
          - log_ratios: [N_test] integrated log ratio.

        Args:
            xs: Test samples, shape [N_test, D].

        Returns:
            torch.Tensor: Log density ratio log(p0(x) / p1(x)), shape [N_test].
                         Returned on CPU device.

        Raises:
            RuntimeError: If model is None (not trained).
        """
        if self.model is None:
            raise RuntimeError(
                "TSM model is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()
        xs = xs.float().to(self.device)

        with torch.no_grad():
            # build uniform tau grid from eps to 1.0
            ts = torch.linspace(
                self.eps, 1.0, self.integration_steps, device=self.device
            )

            # evaluate score at each tau; stack results
            vals = []
            for t in ts:
                tau_broadcast = torch.full(
                    (xs.shape[0], 1), t.item(), device=self.device
                )
                score_t = self.model(xs, tau_broadcast).squeeze(-1)  # [N_test]
                vals.append(-score_t)  # negate for integrand

            vals = torch.stack(vals, dim=0)  # [T, N_test]

            # integrate via trapezoid rule
            dt = (1.0 - self.eps) / (self.integration_steps - 1)
            log_ratios = torch.trapezoid(vals, dx=dt, dim=0).cpu()  # [N_test]

        return log_ratios
