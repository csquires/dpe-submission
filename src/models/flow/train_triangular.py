"""DEPRECATED: train_triangular_flow has been replaced by train_loop + tri_fm_loss."""
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def train_triangular_flow(
    model: nn.Module,
    samples_p0: Tensor,
    samples_p1: Tensor,
    samples_pstar: Tensor,
    n_epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 2e-3,
    score_weight: float = 1.0,
    eps: float = 0.01,
    device: Optional[str] = None,
    verbose: bool = False,
    log_every: int = 100,
    adam_betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.0,
    cosine_min_factor: float = 1.0,
    triangular_p_uncond: float = 0.0,
) -> nn.Module:
    """DEPRECATED: prefer `train_loop(..., tri_fm_loss, ...)`.

    trains a (v, s) flow-matching head on stratified batches from (p_0, p_1, p_*)
    with the score-MSE masked off the p_* rows. uses linear interpolation
    x_t = (1-t) z + t x_data with tau ~ U[eps, 1-eps].
    """
    warnings.warn(
        "train_triangular_flow is deprecated; migrate to train_loop(..., tri_fm_loss, ...).",
        DeprecationWarning,
        stacklevel=2,
    )

    # resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # move model and all three sample tensors to device
    model = model.to(device)
    samples_p0 = samples_p0.to(device)
    samples_p1 = samples_p1.to(device)
    samples_pstar = samples_pstar.to(device)

    # extract shapes
    N0 = samples_p0.shape[0]
    N1 = samples_p1.shape[0]
    Nstar = samples_pstar.shape[0]
    D = samples_p0.shape[1]

    # validate: nonempty samples
    if N0 == 0 or N1 == 0 or Nstar == 0:
        raise ValueError("all three sample tensors must be nonempty")

    # validate: feature dimension match
    if D != samples_p1.shape[1] or D != samples_pstar.shape[1]:
        raise ValueError("samples_p0, samples_p1, samples_pstar must have same feature dimension D")

    # validate: batch size constraint
    if batch_size < 3:
        raise ValueError(f"batch_size must be >= 3, got {batch_size}")

    # set model to train mode
    model.train()

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=tuple(adam_betas), eps=1e-8, weight_decay=float(weight_decay))
    scheduler = (None if cosine_min_factor == 1.0 else
                 optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=n_epochs, eta_min=lr * float(cosine_min_factor)))

    # training loop
    for epoch in range(n_epochs):
        # compute per-class batch sizes (stratified)
        n_each = batch_size // 3
        n_p0 = n_each
        n_p1 = n_each
        n_pstar = batch_size - 2 * n_each  # absorbs remainder

        # sample indices
        idx_p0 = torch.randint(0, N0, (n_p0,), device=device)
        idx_p1 = torch.randint(0, N1, (n_p1,), device=device)
        idx_pstar = torch.randint(0, Nstar, (n_pstar,), device=device)

        # construct mixed batch
        x_data = torch.cat([
            samples_p0[idx_p0],
            samples_p1[idx_p1],
            samples_pstar[idx_pstar],
        ], dim=0)  # [batch_size, D]

        # construct class indices and convert to onehot
        y_idx = torch.cat([
            torch.zeros(n_p0, dtype=torch.long, device=device),
            torch.ones(n_p1, dtype=torch.long, device=device),
            torch.full((n_pstar,), 2, dtype=torch.long, device=device),
        ], dim=0)  # [batch_size]
        y_onehot = F.one_hot(y_idx, num_classes=3).to(x_data.dtype)  # [batch_size, 3]

        # class-conditioning dropout (mirrors p_uncond in train_conditional).
        # with prob triangular_p_uncond, zero the onehot row -> "unconditional"
        # signal to the model; it must learn a class-agnostic scaffold.
        if triangular_p_uncond > 0.0:
            mask = torch.bernoulli(torch.full((batch_size, 1), triangular_p_uncond, device=device))
            y_onehot = torch.where(mask > 0.5, torch.zeros_like(y_onehot), y_onehot)

        # sample noise and time
        z = torch.randn(batch_size, D, device=device)  # [batch_size, D]
        t = torch.rand(batch_size, 1, device=device) * (1 - 2 * eps) + eps  # [batch_size, 1]

        # linear interpolation
        x_t = (1 - t) * z + t * x_data  # [batch_size, D]

        # compute targets
        v_target = x_data - z  # [batch_size, D]
        s_target = -z / (1 - t)  # [batch_size, D]; safe since (1-t) >= eps

        # forward pass
        v_pred, s_pred = model.forward_from_onehot(t, x_t, y_onehot)  # [batch_size, D] each

        # compute losses
        loss_v = F.mse_loss(v_pred, v_target)

        # masked score loss: only p_0 and p_1 contribute
        s_mask = (y_idx != 2).to(x_data.dtype).unsqueeze(-1)  # [batch_size, 1]
        diff_s = (s_pred - s_target) ** 2  # [batch_size, D]
        diff_s = diff_s * s_mask  # zero out p_* rows
        n_active = s_mask.sum() * D  # total nonzero entries after broadcast
        loss_s = diff_s.sum() / n_active.clamp(min=1.0)

        loss_total = loss_v + score_weight * loss_s

        # optimization step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # verbose logging
        if verbose and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch+1}: loss_v={loss_v.item():.4f}, loss_s={loss_s.item():.4f}, loss_total={loss_total.item():.4f}")

    # set model to eval mode
    model.eval()

    return model
