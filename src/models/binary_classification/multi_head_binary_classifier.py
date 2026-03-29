import torch
import torch.nn as nn
from typing import List


class MultiHeadBinaryClassifier(nn.Module):
    """Multi-head binary classifier with shared backbone and independent heads.

    Processes samples through a shared feature extraction backbone, then applies
    independent 2-layer binary classification heads. Each head discriminates between
    adjacent waypoint distributions in density ratio estimation.

    architecture:
    - backbone: shared feature extractor (configurable depth)
    - heads: num_heads independent MLPs [Linear -> ReLU -> Linear]

    forward(x) -> [batch, num_heads] logits
    fit(xs_per_head, ys_per_head) -> trains backbone + heads jointly
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
        """Initialize multi-head binary classifier.

        args:
        - input_dim: feature dimension of input samples
        - num_heads: number of independent binary classifiers
        - hidden_dim: width of backbone layers (default 10)
        - head_dim: width of intermediate layer in each head (default 10)
        - num_shared_layers: depth of backbone, minimum 1 (default 2)
        - learning_rate: AdamW learning rate (default 0.005)
        - num_epochs: training epochs per fit() call (default 300)

        architecture:
        - backbone: [Linear(input_dim, hidden_dim) -> ReLU] ->
                    [Linear(hidden_dim, hidden_dim) -> ReLU] * (num_shared_layers - 1)
        - heads: nn.ModuleList of num_heads, each [Linear(hidden_dim, head_dim) ->
                 ReLU -> Linear(head_dim, 1)]
        """
        super().__init__()

        # build backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(input_dim, hidden_dim))
        backbone_layers.append(nn.ReLU())
        for _ in range(num_shared_layers - 1):
            backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone_layers)

        # build heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            head = nn.Sequential(
                nn.Linear(hidden_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, 1),
            )
            self.heads.append(head)

        # store config and training hyperparameters
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # initialize weights
        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for all heads in parallel.

        input: x [batch_size, input_dim]
        process:
        1. pass x through backbone -> [batch_size, hidden_dim]
        2. apply each head to extract [batch_size, 1] logits
        3. stack along dimension 1
        output: [batch_size, num_heads]
        """
        # backbone: [batch, input_dim] -> [batch, hidden_dim]
        shared = self.backbone(x)

        # apply each head and collect outputs
        head_outputs = []
        for head in self.heads:
            # head[i]: [batch, hidden_dim] -> [batch, 1]
            logits_i = head(shared)
            head_outputs.append(logits_i)

        # stack along dimension 1: [batch, num_heads]
        output = torch.cat(head_outputs, dim=1)
        return output

    def _reset_parameters(self) -> None:
        """Initialize all weights to xavier uniform, biases to zero."""
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for head in self.heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def fit(
        self,
        xs_per_head: List[torch.Tensor],
        ys_per_head: List[torch.Tensor],
    ) -> None:
        """Train backbone and all heads jointly on independent tasks.

        inputs:
        - xs_per_head: list of length num_heads, each element [n_i, input_dim]
          (n_i can vary across heads)
        - ys_per_head: list of length num_heads, each element [n_i, 1] with
          values in {0.0, 1.0}

        process:
        1. reset all weights
        2. set training mode
        3. create loss function and optimizer
        4. for each epoch:
           a. zero gradients
           b. for each head: forward pass, compute loss (accumulate)
           c. average loss across heads
           d. backward and step
        5. set eval mode after training
        """
        self._reset_parameters()
        self.train()

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        num_heads = len(xs_per_head)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            losses = []
            for i in range(num_heads):
                # forward: [n_i, input_dim] -> [n_i, num_heads]
                logits = self.forward(xs_per_head[i])
                # select head i: [n_i]
                logits_i = logits[:, i]
                # compute loss: [n_i] vs [n_i]
                loss_i = loss_fn(logits_i, ys_per_head[i].squeeze(1))
                losses.append(loss_i)

            # average loss across heads
            total_loss = sum(losses) / num_heads
            total_loss.backward()
            optimizer.step()

        self.eval()

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        """Inference-mode prediction of logits for all heads.

        input: xs [batch_size, input_dim]
        process:
        1. set eval mode
        2. enter no_grad context
        3. forward pass -> [batch_size, num_heads]
        output: [batch_size, num_heads]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(xs)
        return logits


def make_multi_head_binary_classifier(**kwargs) -> MultiHeadBinaryClassifier:
    """Factory function for convenient instantiation.

    args: arbitrary keyword arguments passed to MultiHeadBinaryClassifier
    returns: MultiHeadBinaryClassifier instance
    """
    return MultiHeadBinaryClassifier(**kwargs)
