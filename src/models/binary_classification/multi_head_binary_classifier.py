import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Any

from src.methods.common._report import _make_report


class MultiHeadBinaryClassifier(nn.Module):
    """Multi-head binary classifier with shared backbone and parallel heads.

    Processes samples through a shared feature extraction backbone, then applies
    independent 2-layer binary classification heads IN PARALLEL using batched
    matrix operations. Each head discriminates between adjacent waypoint
    distributions in density ratio estimation.

    architecture:
    - backbone: shared feature extractor (configurable depth)
    - heads: batched parameters for parallel computation
      layer1: [num_heads, hidden_dim, head_dim] + bias [num_heads, head_dim]
      layer2: [num_heads, head_dim, 1] + bias [num_heads, 1]

    forward(x) -> [batch, num_heads] logits (parallel head computation)
    fit(xs_per_head, ys_per_head) -> trains with batched backbone pass
    predict_logits(xs) -> [batch, num_heads] logits in eval mode

    training budget scaling:
    - epoch_scale: multiplies num_epochs to match separate-classifier budget
      (typically set to num_heads to match TriangularTDRE optimization steps)
    - lr_scale: scales learning_rate by 1/sqrt(hidden_dim/base_dim) for stability
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        hidden_dim: int = 10,
        head_dim: int = 10,
        num_shared_layers: int = 2,
        learning_rate: float = 0.005,
        num_epochs: int = 300,
        epoch_scale: int = 1,
        lr_hidden_dim_scale: bool = False,
        lr_base_dim: int = 16,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()

        # build backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(input_dim, hidden_dim))
        backbone_layers.append(nn.ReLU())
        for _ in range(num_shared_layers - 1):
            backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone_layers)

        # batched head parameters for parallel computation
        # layer 1: [num_heads, hidden_dim, head_dim]
        self.heads_w1 = nn.Parameter(torch.empty(num_heads, hidden_dim, head_dim))
        self.heads_b1 = nn.Parameter(torch.empty(num_heads, head_dim))
        # layer 2: [num_heads, head_dim, 1]
        self.heads_w2 = nn.Parameter(torch.empty(num_heads, head_dim, 1))
        self.heads_b2 = nn.Parameter(torch.empty(num_heads, 1))

        # store config
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.batch_size = batch_size

        # apply epoch scaling (match total optimization budget of separate classifiers)
        self.num_epochs = num_epochs * epoch_scale

        # apply learning rate scaling for larger models
        # scale UP for larger hidden_dim to compensate for multi-task gradient averaging
        if lr_hidden_dim_scale:
            import math
            scale_factor = math.sqrt(hidden_dim / lr_base_dim)
            self.learning_rate = learning_rate * scale_factor
        else:
            self.learning_rate = learning_rate

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize backbone with xavier, heads with scaled init."""
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # init heads: xavier-like scaling
        nn.init.xavier_uniform_(self.heads_w1.view(self.num_heads, -1))
        nn.init.zeros_(self.heads_b1)
        nn.init.xavier_uniform_(self.heads_w2.view(self.num_heads, -1))
        nn.init.zeros_(self.heads_b2)

    def _apply_heads_parallel(self, features: torch.Tensor) -> torch.Tensor:
        """Apply all heads in parallel using batched matmul.

        features: [batch, hidden_dim]
        returns: [batch, num_heads]
        """
        # layer 1: features @ w1 + b1
        # features: [batch, hidden_dim] -> [batch, 1, hidden_dim]
        # w1: [num_heads, hidden_dim, head_dim]
        # result: [batch, num_heads, head_dim]
        h = torch.einsum('bh,nhd->bnd', features, self.heads_w1) + self.heads_b1

        # relu
        h = F.relu(h)

        # layer 2: h @ w2 + b2
        # h: [batch, num_heads, head_dim]
        # w2: [num_heads, head_dim, 1]
        # result: [batch, num_heads, 1] -> squeeze -> [batch, num_heads]
        logits = torch.einsum('bnd,ndo->bno', h, self.heads_w2) + self.heads_b2
        return logits.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for all heads in parallel.

        x: [batch, input_dim]
        returns: [batch, num_heads]
        """
        features = self.backbone(x)
        return self._apply_heads_parallel(features)

    def fit(
        self,
        xs_per_head: List[torch.Tensor],
        ys_per_head: List[torch.Tensor],
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_fn: Callable[[Any], torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """train shared backbone + per-head binary classifiers.

        single backbone pass per optimizer step on concatenated head data;
        per-head losses summed (no averaging, see below) and back-propagated.

        if self.batch_size is None or >= max(n_i), runs full-batch (legacy).
        otherwise: per epoch, shuffle each head's indices independently and step
        through mini-batches of size self.batch_size sampled from EACH head;
        cyclic wrap-around when a head's permutation is exhausted within an epoch
        (preserves per-head loss balance even with unequal n_i).

        backprop without averaging: each head needs full gradient signal
        (averaging by num_heads would undertrain individual heads).

        xs_per_head: list of [n_i, input_dim] tensors
        ys_per_head: list of [n_i, 1] tensors with values in {0, 1}
        step_cb: optional callback invoked at step_cb_interval multiples (step, score).
        eval_fn: optional function to compute evaluation score from model.
        step_cb_interval: interval (in steps) between callback invocations.
        """
        self._reset_parameters()
        self.train()

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        num_heads = len(xs_per_head)
        n_per_head = [xs.shape[0] for xs in xs_per_head]
        max_n = max(n_per_head)
        bs = self.batch_size if (self.batch_size and self.batch_size < max_n) else max_n

        # construct reporting closure before training loop
        do_report = _make_report(step_cb, step_cb_interval, eval_fn, self, self)

        def _step(xs_list, ys_list, sizes):
            optimizer.zero_grad()
            xs_cat = torch.cat(xs_list, dim=0)
            features_cat = self.backbone(xs_cat)
            features_split = torch.split(features_cat, sizes, dim=0)
            total_loss = 0.0
            for i, (feat_i, ys_i) in enumerate(zip(features_split, ys_list)):
                h = feat_i @ self.heads_w1[i] + self.heads_b1[i]
                h = F.relu(h)
                logits_i = (h @ self.heads_w2[i] + self.heads_b2[i]).squeeze(-1)
                total_loss = total_loss + loss_fn(logits_i, ys_i.squeeze(-1))
            total_loss.backward()
            optimizer.step()

        if bs == max_n:
            for _ in range(self.num_epochs):
                _step(xs_per_head, ys_per_head, n_per_head)
                do_report()
        else:
            for _ in range(self.num_epochs):
                perms = [
                    torch.randperm(n_per_head[i], device=xs_per_head[i].device)
                    for i in range(num_heads)
                ]
                for start in range(0, max_n, bs):
                    xs_batch, ys_batch = [], []
                    for i in range(num_heads):
                        n_i = n_per_head[i]
                        # cyclic wrap so each head sees bs samples per step even
                        # when start+bs exceeds n_i (preserves per-head loss balance).
                        idx = torch.arange(start, start + bs, device=perms[i].device) % n_i
                        idx = perms[i][idx]
                        xs_batch.append(xs_per_head[i][idx])
                        ys_batch.append(ys_per_head[i][idx])
                    _step(xs_batch, ys_batch, [bs] * num_heads)
                    do_report()

        self.eval()

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        """Inference: all heads applied in parallel.

        xs: [batch, input_dim]
        returns: [batch, num_heads]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(xs)


def make_multi_head_binary_classifier(**kwargs) -> MultiHeadBinaryClassifier:
    """Factory function for convenient instantiation.

    args: arbitrary keyword arguments passed to MultiHeadBinaryClassifier
    returns: MultiHeadBinaryClassifier instance
    """
    return MultiHeadBinaryClassifier(**kwargs)
