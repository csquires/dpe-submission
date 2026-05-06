"""pure function for training 3-class triangular flow matching models."""
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
    """train 3-class triangular flow matching with masked score loss.

    trains model with stratified batches from three distributions:
    p_0 (source), p_1 (target), p_* (auxiliary).
    uses linear interpolation + uniform time sampling in [eps, 1-eps].
    jointly optimizes velocity MSE and masked score MSE.

    score loss only applies to p_0 and p_1 rows; p_* contributes zero
    to score gradient (by masking). rationale: s_t(.|c_*) is never
    queried at inference (only u_t(.|c_*), s_t(.|c_0), s_t(.|c_1) appear
    in the s2 ratio ode). training s_t(.|c_*) would only inject gradient
    noise into the shared backbone in directions irrelevant to the
    objective.

    model interface: forward_from_onehot(t: [B,1], x: [B,D], y_onehot: [B,K]) ->
      (velocity: [B,D], score: [B,D])
    where K=3: index 0=p_0, 1=p_1, 2=p_*.

    args:
      model: nn.Module with forward_from_onehot(t, x, y_onehot) -> (v, s).
             both outputs [B,D] tensors.
      samples_p0: [N_0, D] tensor from distribution p_0.
      samples_p1: [N_1, D] tensor from distribution p_1.
      samples_pstar: [N_*, D] tensor from distribution p_*.
      n_epochs: number of training epochs. default 1000.
      batch_size: total batch size per epoch; split 1/3 per class.
          default 512. must be >= 3.
      lr: learning rate for Adam. default 2e-3.
      score_weight: coefficient for score loss term.
          overall loss = loss_v + score_weight * loss_s.
          default 1.0.
      eps: time clamp to [eps, 1-eps] to avoid t=0 singularity.
          default 0.01.
      device: device string ('cuda', 'cpu'). if None, defaults to
          'cuda' if available, else 'cpu'.
      verbose: if True, print losses every log_every epochs.
      log_every: epoch interval for logging. default 100.

    returns:
      model in eval mode, moved to device, with trained parameters.

    validation (top of function):
      - all three sample tensors nonempty (shape[0] > 0).
      - all three have same feature dimension D (shape[1] match).
      - batch_size >= 3 (at least one sample per class).

    pseudocode:
      1. resolve device (default cuda if available, else cpu).
      2. move model and all three sample tensors to device.
      3. extract N_0, N_1, N_*, D from sample shapes.
      4. validate: N_0 > 0, N_1 > 0, N_* > 0; all D match; batch_size >= 3.
      5. set model.train().
      6. init optimizer = Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8).
      7. for epoch in range(n_epochs):
           a. compute per-class batch sizes:
              n_0 = batch_size // 3
              n_1 = batch_size // 3
              n_* = batch_size - 2 * (batch_size // 3)  (absorbs remainder)
           b. sample indices: idx_p0, idx_p1, idx_pstar.
           c. construct x_data = cat([p0_sampled, p1_sampled, pstar_sampled]).
           d. construct y_idx = cat([zeros(n_0), ones(n_1), 2*ones(n_*)]).
           e. convert y_idx to onehot [B, 3].
           f. sample z ~ N(0, I) same shape as x_data.
           g. sample t ~ U[eps, 1-eps].
           h. x_t = (1 - t) * z + t * x_data.
           i. v_target = x_data - z.
           j. s_target = -z / (1 - t).
           k. forward: v_pred, s_pred = model.forward_from_onehot(t, x_t, y_onehot).
           l. loss_v = MSE(v_pred, v_target)  (unmasked, all B,D).
           m. masked score loss:
              - mask = (y_idx != 2).unsqueeze(-1)  [B, 1] -> broadcasts to [B, D]
              - diff_s = (s_pred - s_target)^2  [B, D]
              - diff_s = diff_s * mask  (zeros out p_* rows)
              - n_active = mask.sum() * D  (count of nonzero entries)
              - loss_s = diff_s.sum() / n_active.clamp(min=1.0)
           n. loss_total = loss_v + score_weight * loss_s.
           o. optimizer.zero_grad(), loss_total.backward(), step().
           p. if verbose and (epoch+1) % log_every == 0:
                print(epoch, loss_v, loss_s, loss_total).
      8. model.eval().
      9. return model.

    score loss detail:
      masking ensures p_* rows contribute zero to score gradient. this is
      mathematically equivalent to computing MSE only over the (n_0 + n_1)*D
      entries corresponding to p_0 and p_1. we divide by n_active (the number
      of nonzero entries after masking) for a mean, not a sum.
    """

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
