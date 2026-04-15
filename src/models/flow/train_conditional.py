"""pure function for training conditional flow matching models."""
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
    """train conditional flow matching with joint velocity + score targets.

    trains model with mixed batches from two distributions (p0 and p1).
    uses linear interpolation between noise and data with uniform time sampling
    in [eps, 1-eps]. jointly optimizes velocity MSE and score MSE losses.

    no learning rate scheduler, no EMA. follows simplicity of SpatialVeloDenoiser2
    training pattern: single Adam optimizer, per-epoch batch construction,
    direct loss backprop.

    model interface: forward(t: [B,1], x: [B,D], c: [B,1]) ->
      (velocity: [B,D], score: [B,D])

    args:
      model: nn.Module with forward signature (t, x, c) -> (v_pred, s_pred).
             both outputs [B,D] tensors.
      samples_p0: [N0, D] tensor of data from distribution p0.
      samples_p1: [N1, D] tensor of data from distribution p1.
      n_epochs: number of training epochs (full passes over sampled batches).
      batch_size: total batch size per epoch.
      lr: learning rate for Adam optimizer. default 2e-3.
      score_weight: weight coefficient for score loss term. default 1.0.
          overall loss = loss_v + score_weight * loss_s.
      eps: clamp time to [eps, 1-eps] to avoid t=0 (zero noise) singularity.
           default 0.01.
      device: device string ('cuda', 'cpu'). if None, defaults to 'cuda' if available.
      verbose: if True, print losses every log_every epochs.
      log_every: epoch interval for verbose logging. default 100.
      p_uncond: per-sample probability of replacing condition with sentinel value.
          default 0.0 means no dropout (existing behavior preserved exactly).
      sentinel_cond: the scalar value to substitute when dropping the condition.
          must be distinct from 0.0 and 1.0 (normal condition range). default -1.0.

    returns:
      model in eval mode, moved to device, with trained parameters.

    pseudocode:
      1. resolve device (default cuda if available, else cpu)
      2. move model, samples_p0, samples_p1 to device
      3. set model.train()
      4. initialize optimizer = Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
      5. for epoch in range(n_epochs):
           a. sample n_p0 = batch_size // 2 random indices from p0
           b. sample n_p1 = batch_size - n_p0 indices from p1
           c. construct x_data = cat([p0_sampled, p1_sampled]) # [batch_size, D]
           d. construct c = cat([zeros(n_p0, 1), ones(n_p1, 1)]) # [batch_size, 1]
           d_cfg. if p_uncond > 0: per-sample condition dropout (replace with sentinel_cond)
           e. sample z ~ N(0, I) same shape as x_data # [batch_size, D]
           f. sample t ~ U[eps, 1-eps] # [batch_size, 1]
           g. interpolate x_t = (1 - t) * z + t * x_data # [batch_size, D]
           h. compute velocity target v_target = x_data - z # [batch_size, D]
           i. compute score target s_target = -z / (1 - t) # [batch_size, D]
           j. forward pass v_pred, s_pred = model(t, x_t, c)
           k. loss_v = MSE(v_pred, v_target)
           l. loss_s = MSE(s_pred, s_target)
           m. loss_total = loss_v + score_weight * loss_s
           n. optimizer.zero_grad()
           o. loss_total.backward()
           p. optimizer.step()
           q. if verbose and (epoch+1) % log_every == 0:
                print(epoch, loss_v, loss_s, loss_total)
      6. model.eval()
      7. return model
    """

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
