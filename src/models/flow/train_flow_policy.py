"""exact mle trainer for bounded-support rq-spline flows over pendulum replay streams."""
import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.models.flow.flow_policy_model import FlowPolicyModel

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FlowTrainCfg:
    """configuration for flow policy training."""
    steps: int = 100_000
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.9999
    heldout_frac: float = 0.2
    warmup_frac: float = 0.05
    seed: int = 0
    hidden_dim: int = 128
    n_layers: int = 3
    n_bins: int = 32


def _build_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.05) -> SequentialLR:
    """build warmup + cosine annealing scheduler.

    linear warmup from 1e-6 to full lr over warmup_frac of total_steps,
    then cosine annealing with eta_min=0 for remainder.
    returns SequentialLR chaining both phases.
    """
    warmup_steps = max(1, int(warmup_frac * total_steps))
    decay_steps = total_steps - warmup_steps

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
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


def flow_ckpt_path(replay_path, cfg, cache_dir):
    """content-addressed ckpt path: key = sha256(replay-file sha256 + cfg json).

    folding the replay hash into the key keeps distinct policies (identical
    cfg, different behavior data) from colliding on one checkpoint. single
    source of truth for the ckpt path (step0 resolves through this).

    args:
        replay_path: str. existing replay h5 (its bytes are hashed).
        cfg: FlowTrainCfg.
        cache_dir: str.

    returns:
        (ckpt_path: str, replay_hash16: str).
    """
    cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    with open(replay_path, "rb") as f:
        replay_hash = hashlib.sha256(f.read()).hexdigest()
    key = hashlib.sha256((replay_hash + cfg_json).encode()).hexdigest()[:16]
    return f"{cache_dir}/flow_{key}.pt", replay_hash[:16]


def heldout_indices(n, cfg):
    """deterministic held-out index set for an n-sample replay.

    single source of truth for the train/held-out split: the trainer excludes
    these indices from training and the gate evaluates ONLY on them.
    """
    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(n)
    return indices[int(n * (1 - cfg.heldout_frac)):]


def load_or_train_flow(
    replay_path: str,
    cfg: FlowTrainCfg,
    cache_dir: str,
    device: str,
    force: bool = False
) -> str:
    """train exact mle flow policy on replay data or load cached checkpoint.

    deterministic split via seeded numpy rng; ema parameter averaging;
    periodic logging every 1000/5000 steps; alternating eval (ema vs raw).
    checkpoint idempotency via sha256 cache key.

    args:
        replay_path: path to h5 replay file with states [N,2], actions [N], phase [N].
        cfg: FlowTrainCfg instance with training hyperparameters.
        cache_dir: directory for checkpoint storage.
        device: torch device string (e.g., 'cpu', 'cuda').
        force: if True, retrain even if checkpoint exists.

    returns:
        checkpoint path (string).
    """

    # phase 0: content-addressed cache key & early return (see flow_ckpt_path)
    cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    ckpt_path, replay_hash = flow_ckpt_path(replay_path, cfg, cache_dir)

    if not force and Path(ckpt_path).exists():
        return ckpt_path

    # phase 1: load replay & split
    with h5py.File(replay_path, "r") as f:
        states = np.array(f["states"], dtype=np.float32)  # [N, 2]
        actions = np.array(f["actions"], dtype=np.float32)  # [N]
        phase = np.array(f["phase"], dtype=np.float32)  # [N]

    N = len(states)

    # deterministic split (heldout_indices is the single source of truth)
    if N < 1e4:
        logger.warning(f"replay has {N} < 10000 samples; gate will reject if heldout < 10000")

    heldout_idx = heldout_indices(N, cfg)
    train_idx = np.setdiff1d(np.arange(N), heldout_idx)

    # phase 2: initialize model & optimizer
    model = FlowPolicyModel(
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_bins=cfg.n_bins
    )
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg.steps, cfg.warmup_frac)

    # phase 3: training loop (all randomness seeded for reproducible ckpts)
    torch.manual_seed(cfg.seed)
    batch_rng = np.random.default_rng(cfg.seed)

    # initialize ema
    ema_state_dict = {name: p.data.clone() for name, p in model.named_parameters()}

    train_nll_history = []
    heldout_nll_history = []

    for step in range(cfg.steps):
        # sample batch
        batch_indices = batch_rng.choice(len(train_idx), size=cfg.batch_size, replace=True)
        s_batch = torch.tensor(states[train_idx[batch_indices]], dtype=torch.float32, device=device)  # [batch_size, 2]
        a_batch = torch.tensor(actions[train_idx[batch_indices]], dtype=torch.float32, device=device)  # [batch_size]

        # forward & loss
        log_prob = model.log_prob(a_batch, s_batch)  # [batch_size]
        loss = -log_prob.mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ema update
        for name, p in model.named_parameters():
            ema_state_dict[name].mul_(cfg.ema_decay).add_(p.data, alpha=1.0 - cfg.ema_decay)

        # logging (train)
        if step % 1000 == 0:
            train_nll_history.append({"step": step, "loss": loss.item()})

        # eval (held-out)
        if step % 5000 == 0:
            # temporarily swap to ema params
            raw_params = {name: p.data.clone() for name, p in model.named_parameters()}
            for name, p in model.named_parameters():
                p.data.copy_(ema_state_dict[name])

            with torch.no_grad():
                s_test = torch.tensor(states[heldout_idx], dtype=torch.float32, device=device)  # [n_heldout, 2]
                a_test = torch.tensor(actions[heldout_idx], dtype=torch.float32, device=device)  # [n_heldout]
                log_prob_test = model.log_prob(a_test, s_test).mean()  # scalar

            heldout_nll_history.append({"step": step, "loss": (-log_prob_test).item()})

            # swap back to raw params
            for name, p in model.named_parameters():
                p.data.copy_(raw_params[name])

    # phase 4: post-training & checkpoint
    # copy ema back to model
    for name, p in model.named_parameters():
        p.data.copy_(ema_state_dict[name])

    model.eval()

    # atomic write
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "cfg": dataclasses.asdict(cfg),
        "replay_hash": replay_hash,
        "split_seed": cfg.seed,
        "train_nll_history": train_nll_history,
        "heldout_nll_history": heldout_nll_history,
        "eval_dtype": "float64"
    }

    tmp_path = ckpt_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)

    return ckpt_path


def load_flow(ckpt_path: str, device: str) -> FlowPolicyModel:
    """load trained flow policy from checkpoint.

    rebuilds model from cfg dict in checkpoint; loads state_dict;
    returns in eval mode on specified device.

    args:
        ckpt_path: path to checkpoint file.
        device: torch device string.

    returns:
        FlowPolicyModel in eval mode.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # trusted local artifact

    cfg_dict = ckpt["cfg"]
    model = FlowPolicyModel(
        hidden_dim=cfg_dict["hidden_dim"],
        n_layers=cfg_dict["n_layers"],
        n_bins=cfg_dict["n_bins"]
    )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return model
