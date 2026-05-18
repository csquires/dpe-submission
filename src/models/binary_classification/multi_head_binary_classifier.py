import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Any

from src.methods.common._report import _make_report


class MultiHeadBinaryClassifier(nn.Module):
    """Multi-head binary classifier with shared backbone and parallel heads.

    Processes samples through a shared feature extraction backbone, then applies
    independent binary classification heads IN PARALLEL using batched matrix
    operations. Each head discriminates between adjacent waypoint distributions
    in density ratio estimation.

    architecture:
    - backbone: num_shared_layers hidden layers (input->hidden, then hidden->hidden,
      ReLU after each).
    - heads: variable-depth batched parameters for parallel computation.
      total hidden layers per head pathway = n_hidden_layers; head-specific hidden
      layers h = n_hidden_layers - num_shared_layers.
      if h >= 1: head has h+1 layers (h hidden layers hidden_dim/head_dim, then
        output head_dim->1).
      if h == 0: head is a single output layer hidden_dim->1.

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
        n_hidden_layers: int = 3,
        learning_rate: float = 0.005,
        num_epochs: int = 300,
        epoch_scale: int = 1,
        lr_hidden_dim_scale: bool = False,
        lr_base_dim: int = 16,
        batch_size: int | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()

        # validation (mirror OrthrosNet)
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        if num_shared_layers < 1:
            raise ValueError("num_shared_layers must be >= 1")
        if num_shared_layers > n_hidden_layers:
            raise ValueError("num_shared_layers must be <= n_hidden_layers")

        h = n_hidden_layers - num_shared_layers  # head-specific hidden layers

        # build backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(input_dim, hidden_dim))
        backbone_layers.append(nn.ReLU())
        for _ in range(num_shared_layers - 1):
            backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone_layers)

        # build batched head parameters
        # h >= 1: h+1 layers; h == 0: single output layer
        head_weights = []
        head_biases = []
        if h >= 1:
            head_weights.append(nn.Parameter(torch.empty(num_heads, hidden_dim, head_dim)))  # [n, h_in, h_dim]
            head_biases.append(nn.Parameter(torch.empty(num_heads, head_dim)))               # [n, h_dim]
            for _ in range(h - 1):
                head_weights.append(nn.Parameter(torch.empty(num_heads, head_dim, head_dim)))  # [n, h_dim, h_dim]
                head_biases.append(nn.Parameter(torch.empty(num_heads, head_dim)))             # [n, h_dim]
            head_weights.append(nn.Parameter(torch.empty(num_heads, head_dim, 1)))  # [n, h_dim, 1]
            head_biases.append(nn.Parameter(torch.empty(num_heads, 1)))             # [n, 1]
        else:
            head_weights.append(nn.Parameter(torch.empty(num_heads, hidden_dim, 1)))  # [n, h_in, 1]
            head_biases.append(nn.Parameter(torch.empty(num_heads, 1)))               # [n, 1]

        self.head_weights = nn.ParameterList(head_weights)
        self.head_biases = nn.ParameterList(head_biases)

        # store config
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self._h = h  # head-specific hidden layers

        # apply epoch scaling
        self.num_epochs = num_epochs * epoch_scale

        # apply learning rate scaling for larger models
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

        # xavier per head weight, zeros per head bias
        for W in self.head_weights:
            nn.init.xavier_uniform_(W.view(self.num_heads, -1))
        for b in self.head_biases:
            nn.init.zeros_(b)

    def _apply_heads_parallel(self, features: torch.Tensor) -> torch.Tensor:
        """Apply all heads in parallel using batched einsum.

        features [batch, hidden_dim] -> layer0 (first einsum uses 'bh,nho->bno')
        -> relu -> layer1..h-1 ('bnh,nho->bno') -> relu -> output layer -> squeeze.
        relu applied between layers, not after the final output layer.

        features: [batch, hidden_dim]
        returns: [batch, num_heads]
        """
        # first layer: features [batch, hidden_dim] x W[0] [num_heads, hidden_dim, out]
        x = torch.einsum('bh,nho->bno', features, self.head_weights[0]) + self.head_biases[0]  # [batch, num_heads, out0]

        num_layers = len(self.head_weights)
        for k in range(1, num_layers):
            x = F.relu(x)
            x = torch.einsum('bnh,nho->bno', x, self.head_weights[k]) + self.head_biases[k]  # [batch, num_heads, outk]

        return x.squeeze(-1)  # [batch, num_heads]

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
                # first head layer: feat_i [n_i, hidden_dim]
                x = feat_i @ self.head_weights[0][i] + self.head_biases[0][i]  # [n_i, out0]
                num_layers = len(self.head_weights)
                for k in range(1, num_layers):
                    x = F.relu(x)
                    x = x @ self.head_weights[k][i] + self.head_biases[k][i]  # [n_i, outk]
                logits_i = x.squeeze(-1)
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


def make_multi_head_binary_classifier(weight_decay: float = 0.0, **kwargs) -> MultiHeadBinaryClassifier:
    """Factory function for convenient instantiation.

    args: arbitrary keyword arguments passed to MultiHeadBinaryClassifier
    returns: MultiHeadBinaryClassifier instance
    """
    return MultiHeadBinaryClassifier(weight_decay=weight_decay, **kwargs)
