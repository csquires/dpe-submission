import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiClassVelScoreMLP(nn.Module):
    """
    multi-class conditional velocity and score MLP.

    architecture (n_shared_layers of n_hidden_layers in the shared trunk):
      one-hot labels [B] -> [B, num_classes]
      input concatenation [t, x, y_onehot] -> [B, D + 1 + K]
        -> backbone: n_shared_layers x [Linear, (opt LN), GELU, (opt LN)]
        -> v_head / s_head: each
              (n_hidden_layers - n_shared_layers) x [Linear, (opt LN), GELU, (opt LN)]
              + Linear(hidden_dim, D)
      output: tuple (velocity [B, D], score [B, D])

    when n_shared_layers == n_hidden_layers the head reduces to a single
    Linear projection (byte-identical to the pre-split MultiClassVelScoreMLP,
    same self.v_head / self.s_head module names and parameter init order).

    two entry points:
      - forward: dispatches from class indices (long) to forward_from_onehot.
      - forward_from_onehot: vmap-safe, takes pre-computed one-hot labels.

    class-index convention (enforced by trainer and ODE driver, not this network):
      for K=3 triangular use, indices map as 0 -> p_0, 1 -> p_1, 2 -> p_*.
      this MLP is K-general; convention is external.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        layernorm: str = "off",
        n_shared_layers: int = 3,
    ) -> None:
        """
        initialize MultiClassVelScoreMLP.

        args:
            input_dim: dimensionality D of data samples.
            num_classes: number of classes K (labels in 0..K-1).
            hidden_dim: width of hidden layers (default 256).
            n_hidden_layers: total hidden Linear+GELU rounds across backbone +
                each head (default 3). final Linear output projection is not
                counted.
            layernorm: layernorm mode in {"off", "pre", "post"}; "pre"/"post"
                insert LayerNorm before/after each hidden GELU in both
                backbone and head hidden rounds (default "off",
                byte-identical to pre-S7).
            n_shared_layers: hidden rounds in the shared backbone; the remaining
                n_hidden_layers - n_shared_layers rounds live in each head
                before the output projection. must satisfy
                1 <= n_shared_layers <= n_hidden_layers (default 3 = fully
                shared, matches the pre-split architecture).
        """
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        if not (1 <= n_shared_layers <= n_hidden_layers):
            raise ValueError(
                f"n_shared_layers must satisfy 1 <= n_shared_layers <= n_hidden_layers "
                f"({n_hidden_layers}); got {n_shared_layers}"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.layernorm = layernorm
        self.n_shared_layers = n_shared_layers

        # shared backbone: n_shared_layers Linear+GELU rounds (with optional
        # LayerNorm pre/post each activation). first layer absorbs the
        # [t, x, y_onehot] concatenation. layernorm "off" preserves byte-
        # identical pre-S7 behavior; the bare-Linear-heads branch below preserves
        # byte-identity for the fully-shared default.
        layers = [nn.Linear(input_dim + 1 + num_classes, hidden_dim)]
        if layernorm == "pre":
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        if layernorm == "post":
            layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(n_shared_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm == "pre":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if layernorm == "post":
                layers.append(nn.LayerNorm(hidden_dim))

        self.backbone = nn.Sequential(*layers)

        n_head_hidden = n_hidden_layers - n_shared_layers
        if n_head_hidden == 0:
            self.v_head = nn.Linear(hidden_dim, input_dim)
            self.s_head = nn.Linear(hidden_dim, input_dim)
        else:
            self.v_head = self._build_head(hidden_dim, input_dim, n_head_hidden, layernorm)
            self.s_head = self._build_head(hidden_dim, input_dim, n_head_hidden, layernorm)

    @staticmethod
    def _build_head(hidden_dim: int, output_dim: int, n_head_hidden: int, layernorm: str) -> nn.Sequential:
        """build a per-head Sequential of n_head_hidden Linear+GELU rounds (with
        optional LayerNorm pre/post each activation) plus a final
        Linear(hidden_dim, output_dim) projection."""
        layers: list[nn.Module] = []
        for _ in range(n_head_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm == "pre":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if layernorm == "post":
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

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
