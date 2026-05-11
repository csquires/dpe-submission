"""DEPRECATED: train_conditional_flow has been replaced by train_loop + fm_loss."""
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def train_conditional_flow(
    model: nn.Module,
    samples_p0: Tensor,
    samples_p1: Tensor,
    n_epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 2e-3,
    score_weight: float = 1.0,
    eps: float = 0.01,
    device: Optional[str] = None,
    verbose: bool = False,
    log_every: int = 100,
    p_uncond: float = 0.0,
    sentinel_cond: float = -1.0,
) -> nn.Module:
    """DEPRECATED: prefer `train_loop(..., fm_loss, ...)`.

    trains a (v, s) flow-matching head on mixed (p0, p1) batches with optional CFG
    dropout. uses linear interpolation x_t = (1-t) z + t x_data with tau ~ U[eps, 1-eps];
    minimises mse(v, x_data - z) + score_weight mse(s, -z / (1 - t)).
    """
    warnings.warn(
        "train_conditional_flow is deprecated; migrate to train_loop(..., fm_loss, ...).",
        DeprecationWarning,
        stacklevel=2,
    )

    # resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # move tensors and model to device
    model = model.to(device)
    samples_p0 = samples_p0.to(device)
    samples_p1 = samples_p1.to(device)

    N0 = samples_p0.shape[0]
    N1 = samples_p1.shape[0]
    D = samples_p0.shape[1]

    # set model to train mode
    model.train()

    # initialize optimizer with explicit eps parameter
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # training loop over epochs
    for epoch in range(n_epochs):
        # sample batch indices from p0 and p1
        n_p0 = batch_size // 2
        n_p1 = batch_size - n_p0

        idx_p0 = torch.randint(0, N0, (n_p0,), device=device)  # [n_p0]
        idx_p1 = torch.randint(0, N1, (n_p1,), device=device)  # [n_p1]

        # construct mixed batch with class labels
        x_p0 = samples_p0[idx_p0]  # [n_p0, D]
        x_p1 = samples_p1[idx_p1]  # [n_p1, D]
        x_data = torch.cat([x_p0, x_p1], dim=0)  # [batch_size, D]

        c = torch.cat([
            torch.zeros(n_p0, 1, device=device),  # class 0 for p0
            torch.ones(n_p1, 1, device=device),   # class 1 for p1
        ], dim=0)  # [batch_size, 1]

        # cfg dropout: per-sample condition masking
        if p_uncond > 0.0:
            mask = torch.bernoulli(torch.full((batch_size, 1), p_uncond, device=device))
            c = torch.where(mask > 0.5, torch.full_like(c, sentinel_cond), c)

        # sample noise
        z = torch.randn(batch_size, D, device=device)  # [batch_size, D]

        # sample time uniformly in [eps, 1-eps]
        t = torch.rand(batch_size, 1, device=device) * (1 - 2*eps) + eps  # [batch_size, 1]

        # linear interpolation
        x_t = (1 - t) * z + t * x_data  # [batch_size, D]

        # compute targets
        v_target = x_data - z  # [batch_size, D]
        s_target = -z / (1 - t)  # [batch_size, D], safe since (1-t) >= eps

        # forward pass
        v_pred, s_pred = model(t, x_t, c)  # v_pred, s_pred: [batch_size, D]

        # compute losses
        loss_v = F.mse_loss(v_pred, v_target)
        loss_s = F.mse_loss(s_pred, s_target)
        loss_total = loss_v + score_weight * loss_s

        # optimization step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # verbose logging
        if verbose and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch+1}: loss_v={loss_v.item():.4f}, loss_s={loss_s.item():.4f}, loss_total={loss_total.item():.4f}")

    # set model to eval mode
    model.eval()

    return model
