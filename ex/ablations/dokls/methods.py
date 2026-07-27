"""dokls ablation: DRE subclasses wrapping step-wise critics.

BDRE_NWJ, BDRE_DV, MHT_NWJ, MHT_DV expose both monolithic fit() and step-wise
init_fit/train_step/finalize for interleaving by TwoLegELDR.
"""
from typing import Callable

import torch

from ex.ablations.dokls.critics import StepBinaryCritic, StepMultiHeadCritic
from ex.ablations.dokls.losses import NWJSpec, DVSpec
from src.methods.common.base import DRE
from src.waypoints.waypoints1d import WaypointBuilder1D, DefaultWaypointBuilder1D


class BDRE_NWJ(DRE):
    """binary DRE with NWJ loss, step-wise and monolithic paths."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        n_hidden_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_steps: int = 3200,
        batch_size: int = 256,
        device: str = "cuda",
        **extra_hp
    ):
        """initialize hyperparameters; defer critic instantiation to init_fit()."""
        super().__init__(input_dim)
        self.latent_dim = latent_dim
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = device
        self.critic = None
        self._step_count = 0

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """monolithic training path.

        procedure:
            - label p0 -> 1, p1 -> 0 (convention: numerator=1, denominator=0)
            - init_fit(samples_p0, samples_p1)
            - loop n_steps times: train_step(), optionally call step_cb
            - finalize()
        """
        self.init_fit(samples_p0, samples_p1)

        for step in range(self.n_steps):
            loss = self.train_step()

            if step_cb is not None and (step + 1) % step_cb_interval == 0:
                if eval_data is not None:
                    # evaluate on pstar samples
                    eval_pstar = eval_data["pstar"]
                    eval_true_ldrs = eval_data["true_ldrs"]
                    predicted = self.predict_ldr(eval_pstar)
                    target = eval_true_ldrs.to(predicted.device)
                    mae = torch.abs(predicted - target).mean().item()
                    step_cb(step, mae)
                else:
                    # no eval_data: pass loss as metric
                    step_cb(step, loss)

        self.finalize()

    def init_fit(self, samples_num: torch.Tensor, samples_den: torch.Tensor) -> None:
        """step-wise initialization: instantiate critic and set up sampler.

        args:
            samples_num: [N, input_dim], numerator group (label=1)
            samples_den: [N, input_dim], denominator group (label=0)
        """
        if self.critic is None:
            self.critic = StepBinaryCritic(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                n_hidden_layers=self.n_hidden_layers,
                loss_spec=NWJSpec(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                device=self.device,
            )
        self.critic.init_fit(samples_num, samples_den)
        self._step_count = 0

    def train_step(self) -> float:
        """one optimizer step; return scalar loss."""
        loss = self.critic.train_step()
        self._step_count += 1
        return loss

    def finalize(self) -> None:
        """post-training finalization (NWJ no-op, DV computes c)."""
        self.critic.finalize()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """log-density ratio: [N] output."""
        return self.critic.predict_ldr(xs)


class BDRE_DV(DRE):
    """binary DRE with DV loss, step-wise and monolithic paths."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        n_hidden_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_steps: int = 3200,
        batch_size: int = 256,
        device: str = "cuda",
        **extra_hp
    ):
        """initialize hyperparameters; defer critic instantiation to init_fit()."""
        super().__init__(input_dim)
        self.latent_dim = latent_dim
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = device
        self.critic = None
        self._step_count = 0

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """monolithic training path.

        procedure:
            - label p0 -> 1, p1 -> 0
            - init_fit(samples_p0, samples_p1)
            - loop n_steps times: train_step(), optionally call step_cb
            - finalize() (computes DV constant c)
        """
        self.init_fit(samples_p0, samples_p1)

        for step in range(self.n_steps):
            loss = self.train_step()

            if step_cb is not None and (step + 1) % step_cb_interval == 0:
                if eval_data is not None:
                    # evaluate on pstar samples
                    eval_pstar = eval_data["pstar"]
                    eval_true_ldrs = eval_data["true_ldrs"]
                    predicted = self.predict_ldr(eval_pstar)
                    target = eval_true_ldrs.to(predicted.device)
                    mae = torch.abs(predicted - target).mean().item()
                    step_cb(step, mae)
                else:
                    # no eval_data: pass loss as metric
                    step_cb(step, loss)

        self.finalize()

    def init_fit(self, samples_num: torch.Tensor, samples_den: torch.Tensor) -> None:
        """step-wise initialization: instantiate critic and set up sampler.

        args:
            samples_num: [N, input_dim], numerator group (label=1)
            samples_den: [N, input_dim], denominator group (label=0)
        """
        if self.critic is None:
            self.critic = StepBinaryCritic(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                n_hidden_layers=self.n_hidden_layers,
                loss_spec=DVSpec(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                device=self.device,
            )
        self.critic.init_fit(samples_num, samples_den)
        self._step_count = 0

    def train_step(self) -> float:
        """one optimizer step; return scalar loss."""
        loss = self.critic.train_step()
        self._step_count += 1
        return loss

    def finalize(self) -> None:
        """post-training finalization: DV computes c from denominator logits."""
        self.critic.finalize()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """log-density ratio: logits - c (DV constant)."""
        return self.critic.predict_ldr(xs)


class MHT_NWJ(DRE):
    """multi-head TDRE with NWJ loss per head, step-wise and monolithic paths."""

    def __init__(
        self,
        input_dim: int,
        num_waypoints: int = 10,
        latent_dim: int = 128,
        n_hidden_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_steps: int = 3200,
        batch_size: int = 256,
        device: str = "cuda",
        waypoint_builder: WaypointBuilder1D | None = None,
        **extra_hp
    ):
        """initialize hyperparameters; defer critic instantiation to init_fit()."""
        super().__init__(input_dim)
        self.num_waypoints = num_waypoints
        self.latent_dim = latent_dim
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = device
        self.waypoint_builder = waypoint_builder or DefaultWaypointBuilder1D()
        self.critic = None
        self._step_count = 0

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """monolithic training path.

        procedure:
            - build waypoints [num_waypoints, batch, input_dim]
            - construct per-head (xs, ys) pairs
            - init_fit()
            - loop n_steps times: train_step(), optionally call step_cb
            - finalize()
        """
        # build waypoints
        waypoints = self.waypoint_builder.build_waypoints(
            samples_p0, samples_p1, self.num_waypoints
        )  # [num_waypoints, batch, input_dim]

        # store for init_fit
        self._waypoints = waypoints

        self.init_fit(samples_p0, samples_p1)

        for step in range(self.n_steps):
            loss = self.train_step()

            if step_cb is not None and (step + 1) % step_cb_interval == 0:
                if eval_data is not None:
                    # evaluate on pstar samples
                    eval_pstar = eval_data["pstar"]
                    eval_true_ldrs = eval_data["true_ldrs"]
                    predicted = self.predict_ldr(eval_pstar)
                    target = eval_true_ldrs.to(predicted.device)
                    mae = torch.abs(predicted - target).mean().item()
                    step_cb(step, mae)
                else:
                    # no eval_data: pass loss as metric
                    step_cb(step, loss)

        self.finalize()

    def init_fit(self, samples_num: torch.Tensor, samples_den: torch.Tensor) -> None:
        """step-wise initialization: build waypoints and instantiate critic.

        args:
            samples_num: [N, input_dim], numerator group (p0 or pstar)
            samples_den: [N, input_dim], denominator group (p1 or p_i)
        """
        # build waypoints if not already done by fit()
        if not hasattr(self, "_waypoints"):
            waypoints = self.waypoint_builder.build_waypoints(
                samples_num, samples_den, self.num_waypoints
            )  # [num_waypoints, batch, input_dim]
        else:
            waypoints = self._waypoints
            delattr(self, "_waypoints")

        # per-head numerator/denominator lists: head i discriminates waypoint i
        # (numerator) from waypoint i+1 (denominator). the critic does its own
        # labeling and stratified sampling from these groups.
        num_heads = self.num_waypoints - 1
        num_list = [waypoints[i] for i in range(num_heads)]
        den_list = [waypoints[i + 1] for i in range(num_heads)]

        # instantiate critic
        if self.critic is None:
            self.critic = StepMultiHeadCritic(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                n_hidden_layers=self.n_hidden_layers,
                num_heads=num_heads,
                loss_specs=[NWJSpec() for _ in range(num_heads)],
                lr=self.lr,
                weight_decay=self.weight_decay,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                device=self.device,
            )

        self.critic.init_fit(num_list, den_list)
        self._step_count = 0

    def train_step(self) -> float:
        """one optimizer step; return scalar loss."""
        loss = self.critic.train_step()
        self._step_count += 1
        return loss

    def finalize(self) -> None:
        """post-training finalization."""
        self.critic.finalize()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """log-density ratio: sum per-head logits minus K (NWJ constant).

        telescoping: sum_k (logits_k - 1) = sum_k logits_k - K.
        """
        logits_per_head = self.critic.predict_logits(xs)  # [N, num_heads]
        num_heads = logits_per_head.shape[1]
        return logits_per_head.sum(dim=1) - num_heads  # [N]


class MHT_DV(DRE):
    """multi-head TDRE with DV loss per head, step-wise and monolithic paths."""

    def __init__(
        self,
        input_dim: int,
        num_waypoints: int = 10,
        latent_dim: int = 128,
        n_hidden_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_steps: int = 3200,
        batch_size: int = 256,
        device: str = "cuda",
        waypoint_builder: WaypointBuilder1D | None = None,
        **extra_hp
    ):
        """initialize hyperparameters; defer critic instantiation to init_fit()."""
        super().__init__(input_dim)
        self.num_waypoints = num_waypoints
        self.latent_dim = latent_dim
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = device
        self.waypoint_builder = waypoint_builder or DefaultWaypointBuilder1D()
        self.critic = None
        self._step_count = 0

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """monolithic training path.

        procedure:
            - build waypoints [num_waypoints, batch, input_dim]
            - construct per-head (xs, ys) pairs
            - init_fit()
            - loop n_steps times: train_step(), optionally call step_cb
            - finalize() (computes per-head DV constants c_k)
        """
        # build waypoints
        waypoints = self.waypoint_builder.build_waypoints(
            samples_p0, samples_p1, self.num_waypoints
        )  # [num_waypoints, batch, input_dim]

        # store for init_fit
        self._waypoints = waypoints

        self.init_fit(samples_p0, samples_p1)

        for step in range(self.n_steps):
            loss = self.train_step()

            if step_cb is not None and (step + 1) % step_cb_interval == 0:
                if eval_data is not None:
                    # evaluate on pstar samples
                    eval_pstar = eval_data["pstar"]
                    eval_true_ldrs = eval_data["true_ldrs"]
                    predicted = self.predict_ldr(eval_pstar)
                    target = eval_true_ldrs.to(predicted.device)
                    mae = torch.abs(predicted - target).mean().item()
                    step_cb(step, mae)
                else:
                    # no eval_data: pass loss as metric
                    step_cb(step, loss)

        self.finalize()

    def init_fit(self, samples_num: torch.Tensor, samples_den: torch.Tensor) -> None:
        """step-wise initialization: build waypoints and instantiate critic.

        args:
            samples_num: [N, input_dim], numerator group (p0 or pstar)
            samples_den: [N, input_dim], denominator group (p1 or p_i)
        """
        # build waypoints if not already done by fit()
        if not hasattr(self, "_waypoints"):
            waypoints = self.waypoint_builder.build_waypoints(
                samples_num, samples_den, self.num_waypoints
            )  # [num_waypoints, batch, input_dim]
        else:
            waypoints = self._waypoints
            delattr(self, "_waypoints")

        # per-head numerator/denominator lists: head i discriminates waypoint i
        # (numerator) from waypoint i+1 (denominator). the critic does its own
        # labeling and stratified sampling from these groups.
        num_heads = self.num_waypoints - 1
        num_list = [waypoints[i] for i in range(num_heads)]
        den_list = [waypoints[i + 1] for i in range(num_heads)]

        # instantiate critic
        if self.critic is None:
            self.critic = StepMultiHeadCritic(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                n_hidden_layers=self.n_hidden_layers,
                num_heads=num_heads,
                loss_specs=[DVSpec() for _ in range(num_heads)],
                lr=self.lr,
                weight_decay=self.weight_decay,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                device=self.device,
            )

        self.critic.init_fit(num_list, den_list)
        self._step_count = 0

    def train_step(self) -> float:
        """one optimizer step; return scalar loss."""
        loss = self.critic.train_step()
        self._step_count += 1
        return loss

    def finalize(self) -> None:
        """post-training finalization: per-head DV constants c_k computed."""
        self.critic.finalize()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """log-density ratio: sum per-head (logits - c_k).

        each head k has its own DV constant c_k set at finalize().
        """
        logits_per_head = self.critic.predict_logits(xs)  # [N, num_heads]
        # subtract per-head DV constant from each head's logits
        ldrs = logits_per_head.clone()
        for k, spec in enumerate(self.critic.loss_specs):
            # access per-head DV constant c_k
            c_k = spec.c
            ldrs[:, k] = logits_per_head[:, k] - c_k
        return ldrs.sum(dim=1)  # [N]
