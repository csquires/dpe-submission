import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


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
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

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
    ) -> None:
        """Train with batched backbone pass and per-head loss.

        Optimizations:
        1. Single backbone pass on concatenated data (O(1) vs O(num_heads))
        2. Parallel head application via einsum
        3. Efficient loss accumulation

        xs_per_head: list of [n_i, input_dim] tensors
        ys_per_head: list of [n_i, 1] tensors with values in {0, 1}
        """
        self._reset_parameters()
        self.train()

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        num_heads = len(xs_per_head)

        # precompute batch sizes for splitting
        batch_sizes = [xs.shape[0] for xs in xs_per_head]

        for _ in range(self.num_epochs):
            optimizer.zero_grad()

            # single backbone pass on all data
            xs_cat = torch.cat(xs_per_head, dim=0)  # [total, input_dim]
            features_cat = self.backbone(xs_cat)    # [total, hidden_dim]

            # split features back by head
            features_split = torch.split(features_cat, batch_sizes, dim=0)

            # compute loss per head (heads applied in parallel per split)
            total_loss = 0.0
            for i, (feat_i, ys_i) in enumerate(zip(features_split, ys_per_head)):
                # apply only head i to its features
                # feat_i: [n_i, hidden_dim]
                h = feat_i @ self.heads_w1[i] + self.heads_b1[i]  # [n_i, head_dim]
                h = F.relu(h)
                logits_i = (h @ self.heads_w2[i] + self.heads_b2[i]).squeeze(-1)  # [n_i]
                total_loss = total_loss + loss_fn(logits_i, ys_i.squeeze(-1))

            # average and backprop
            (total_loss / num_heads).backward()
            optimizer.step()

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
