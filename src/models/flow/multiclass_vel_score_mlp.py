import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiClassVelScoreMLP(nn.Module):
    """
    multi-class conditional velocity and score MLP.

    architecture:
      one-hot labels [B] -> [B, num_classes]
      input concatenation [t, x, y_onehot] -> [B, D + 1 + K]
        -> backbone: 3 hidden layers with GELU
        -> v_head: Linear(hidden_dim, D) [velocity]
        -> s_head: Linear(hidden_dim, D) [score]
      output: tuple (velocity [B, D], score [B, D])

    two entry points:
      - forward: dispatches from class indices (long) to forward_from_onehot.
      - forward_from_onehot: vmap-safe, takes pre-computed one-hot labels.

    class-index convention (enforced by trainer and ODE driver, not this network):
      for K=3 triangular use, indices map as 0 -> p_0, 1 -> p_1, 2 -> p_*.
      this MLP is K-general; convention is external.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, n_hidden_layers: int = 3) -> None:
        """
        initialize MultiClassVelScoreMLP.

        args:
            input_dim: dimensionality D of data samples.
            num_classes: number of classes K (labels in 0..K-1).
            hidden_dim: width of hidden layers (default 256).
            n_hidden_layers: number of hidden layers in backbone (default 3, must be >= 1).

        behavior:
          1. validate n_hidden_layers >= 1.
          2. store input_dim, num_classes, hidden_dim, n_hidden_layers as instance attributes.
          3. build shared backbone as nn.Sequential with n_hidden_layers hidden layers.
             first layer: Linear(D+1+K, hidden_dim) + GELU.
             remaining layers: (n_hidden_layers-1) * [Linear(hidden_dim, hidden_dim) + GELU].
          4. build velocity head: Linear(hidden_dim, D).
          5. build score head: Linear(hidden_dim, D).
        """
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        layers = [nn.Linear(input_dim + 1 + num_classes, hidden_dim), nn.GELU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        self.backbone = nn.Sequential(*layers)

        self.v_head = nn.Linear(hidden_dim, input_dim)
        self.s_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, t: Tensor, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        compute velocity and score from time, data, and class label.

        given time t in [0,1], data x in R^D, and class label y in 0..K-1,
        compute velocity v(t,x,y) and score s(t,x,y).

        procedure:
          1. one-hot encode y.
          2. cast to match x.dtype.
          3. dispatch to forward_from_onehot.

        args:
            t: time [B, 1]
            x: data [B, D]
            y: class indices [B], dtype long, range 0..num_classes-1

        returns:
            (velocity [B, D], score [B, D])

        edge cases:
          - y must be long tensor; F.one_hot raises TypeError otherwise.
          - y must be in [0, num_classes-1]; F.one_hot raises if out of range.
        """
        y_onehot = F.one_hot(y, num_classes=self.num_classes).to(x.dtype)
        return self.forward_from_onehot(t, x, y_onehot)

    def forward_from_onehot(self, t: Tensor, x: Tensor, y_onehot: Tensor) -> tuple[Tensor, Tensor]:
        """
        compute velocity and score from pre-computed one-hot labels.

        given time t in [0,1], data x in R^D, and one-hot labels y_onehot,
        compute velocity v(t,x,y) and score s(t,x,y). no F.one_hot inside;
        vmap-safe entry point.

        procedure:
          1. concatenate [t, x, y_onehot] along dim=-1 -> [B, D+1+K].
          2. pass through backbone -> [B, hidden_dim].
          3. apply velocity head -> [B, D].
          4. apply score head -> [B, D].
          5. return tuple (velocity, score).

        args:
            t: time [B, 1]
            x: data [B, D]
            y_onehot: one-hot labels [B, num_classes], dtype matching x.dtype

        returns:
            (velocity [B, D], score [B, D])

        edge cases:
          - y_onehot must have shape [B, num_classes]; mismatched K surfaces
            as torch.cat shape error.
          - y_onehot dtype must match x.dtype for safe concatenation.
        """
        h = torch.cat([t, x, y_onehot], dim=-1)  # [B, D+1+K]
        h = self.backbone(h)  # [B, hidden_dim]
        v = self.v_head(h)  # [B, D]
        s = self.s_head(h)  # [B, D]
        return v, s


if __name__ == "__main__":
    # smoke test: instantiate, build batch, call both forward entry points,
    # assert output shapes and equivalence.
    model = MultiClassVelScoreMLP(input_dim=5, num_classes=3, hidden_dim=128)

    batch_size = 4
    t = torch.randn(batch_size, 1)
    x = torch.randn(batch_size, 5)
    y = torch.randint(0, 3, (batch_size,))

    # call forward(t, x, y)
    v1, s1 = model(t, x, y)

    # call forward_from_onehot with same batch
    y_onehot = F.one_hot(y, 3).float()
    v2, s2 = model.forward_from_onehot(t, x, y_onehot)

    # assert shapes
    assert v1.shape == (batch_size, 5), f"velocity shape mismatch: {v1.shape}"
    assert s1.shape == (batch_size, 5), f"score shape mismatch: {s1.shape}"
    assert v2.shape == (batch_size, 5), f"velocity (onehot) shape mismatch: {v2.shape}"
    assert s2.shape == (batch_size, 5), f"score (onehot) shape mismatch: {s2.shape}"

    # assert equivalence
    assert torch.allclose(v1, v2), "velocity mismatch between forward and forward_from_onehot"
    assert torch.allclose(s1, s2), "score mismatch between forward and forward_from_onehot"

    print("ok")
