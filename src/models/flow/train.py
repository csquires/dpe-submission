"""pure function for training velocity matching models via flow matching."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def _build_scheduler(optimizer, total_steps: int) -> SequentialLR:
    """build warmup + cosine annealing scheduler.

    5% warmup with LinearLR, then CosineAnnealingLR for remainder.
    returns SequentialLR chaining both phases.
    """
    warmup_steps = max(1, int(0.05 * total_steps))
    decay_steps = total_steps - warmup_steps

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,  # start near zero
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=0.0
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )


def train_flow(
    model: nn.Module,
    latent_codes: Tensor,      # [N, D] raw tensor of training data
    total_steps: int = 250000,
    batch_size: int = 128,
    lr: float = 1e-2,
    device: str = "cuda",
    ckpt_path: str | None = None,
    ema_decay: float = 0.9999,
) -> nn.Module:
    """train velocity matching model via flow matching with ema regularization.

    pure function: loads checkpoint if provided and exists (early return),
    otherwise trains model and returns with ema-averaged parameters.

    checkpoint loading is idempotent: re-runs with same ckpt_path
    will always return the checkpoint state.

    train loop:
      1. sample batch of data indices
      2. sample gaussian noise
      3. sample time from Beta(2, 1) distribution (biased to t=1, data regime)
      4. interpolate: x_t = (1 - t) * z + t * x1
      5. compute target velocity: v = x1 - z
      6. predict velocity: v_pred = model(x_t, t)
      7. loss = mse(v_pred, v)
      8. optimize
      9. update ema parameters
      10. step scheduler

    after training: copy ema params back to model, set eval mode, save checkpoint.

    args:
      model: nn.Module with forward(x_t: [B, D], t: [B,]) -> [B, D]
      latent_codes: [N, D] training data
      total_steps: number of training steps
      batch_size: batch size per step
      lr: learning rate
      device: device to train on
      ckpt_path: path to checkpoint. if exists, load and return immediately.
      ema_decay: decay factor for exponential moving average

    returns:
      model in eval mode, with ema-averaged parameters, moved to device
    """

    # phase 1: checkpoint loading
    if ckpt_path is not None and Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(device)
        model.eval()
        return model

    # phase 2: device setup
    model = model.to(device)
    latent_codes = latent_codes.to(device)
    N = latent_codes.shape[0]
    D = latent_codes.shape[1]

    # phase 3: optimizer and scheduler setup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = _build_scheduler(optimizer, total_steps)

    # phase 4: ema model initialization
    ema_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # phase 5: training loop
    for step in range(total_steps):
        # 5a: batch sampling
        idx = torch.randint(0, N, (batch_size,))
        x1 = latent_codes[idx]  # [batch_size, D]

        # 5b: noise sampling
        z = torch.randn_like(x1)  # [batch_size, D]

        # 5c: time sampling (beta distribution)
        t = torch.distributions.Beta(2.0, 1.0).sample((batch_size,)).to(device)  # [batch_size,]

        # 5d: interpolation
        x_t = (1 - t.unsqueeze(1)) * z + t.unsqueeze(1) * x1  # [batch_size, D]

        # 5e: target velocity
        v_target = x1 - z  # [batch_size, D]

        # 5f: model prediction
        v_pred = model(x_t, t)  # [batch_size, D]

        # 5g: loss computation
        loss = F.mse_loss(v_pred, v_target)

        # 5h: optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 5i: ema update
        for name, param in model.named_parameters():
            ema_params[name].mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)

    # phase 6: post-training
    for name, param in model.named_parameters():
        param.data.copy_(ema_params[name])
    model.eval()

    # phase 7: checkpoint saving
    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    # phase 8: return
    return model
