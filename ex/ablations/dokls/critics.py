"""critic networks for dokls ablation: binary and multi-head density ratio estimators.

architecture follows default_binary_classifier.py and multi_head_binary_classifier.py
conventions. zero-init of final layer for NWJ/DV ensures stable gradient flow at step 0.

stratified minibatch sampling (bs/2 per group) keeps numerator/denominator balanced.
autograd through mask indexing is safe (torch.randint preserves gradients).
"""
import torch
import torch.nn as nn
from abc import ABC


class LossSpec(ABC):
    """abstract loss specification (mirrors ex.ablations.dokls.losses.LossSpec).

    contracts:
    - loss_fn(logits [N], labels [N]) -> scalar tensor
    - predict_ldr(logits [N]) -> [N] tensor
    - finalize(denom_logits [M]) -> None (optional; default no-op)
    """
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_ldr(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def finalize(self, denom_logits: torch.Tensor) -> None:
        pass


class StepBinaryCritic(nn.Module):
    """binary density ratio critic with stratified minibatch training.

    architecture:
        input_dim -> latent_dim (ReLU) -> [latent_dim (ReLU) x (n_hidden_layers-1)] -> 1

    total layers: n_hidden_layers + 1 (input->latent + hidden->hidden x (n_hidden_layers-1) + latent->1).

    for NWJSpec/DVSpec, final layer (latent->1) is zero-initialized to stabilize early training.
    other layers: xavier_uniform weights, zero biases (matching DefaultBinaryClassifier).

    training via stratified sampling: each train_step draws bs/2 samples from numerator (P, label=1)
    and bs/2 from denominator (Q, label=0), passes through loss_spec, and steps the optimizer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_hidden_layers: int,
        loss_spec: LossSpec,
        lr: float,
        weight_decay: float,
        n_steps: int,
        batch_size: int | None,
        device: torch.device,
    ):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")

        # build mlp layers
        layers = []
        layers.append(nn.Linear(input_dim, latent_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent_dim, 1))
        self.model = nn.Sequential(*layers)

        # move to device
        self.to(device)
        self.device = device
        self.loss_spec = loss_spec
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size

        # initialize weights and biases
        self._reset_parameters(loss_spec)

    def _reset_parameters(self, loss_spec: LossSpec) -> None:
        """initialize weights (xavier) and biases (zeros); zero-init final layer for NWJ/DV."""
        final_layer = None
        for i, module in enumerate(self.model):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # track final linear layer
                final_layer = module

        # zero-init final layer if NWJSpec or DVSpec
        if final_layer is not None:
            # check class name to detect NWJ/DV (avoid circular import)
            spec_name = loss_spec.__class__.__name__
            if spec_name in ("NWJSpec", "DVSpec"):
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass. x: [N, input_dim], returns [N, 1]."""
        return self.model(x)

    def init_fit(
        self,
        samples_num: torch.Tensor,  # [M, input_dim], numerator group (p*)
        samples_den: torch.Tensor,  # [M, input_dim], denominator group (p_i)
    ) -> None:
        """store samples and build optimizer.

        samples_num, samples_den: reference (no clone); can differ in size.
        builds self.optimizer with AdamW.
        """
        self.samples_num = samples_num  # [n_num, input_dim]
        self.samples_den = samples_den  # [n_den, input_dim]
        self.n_num = samples_num.shape[0]
        self.n_den = samples_den.shape[0]
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_step(self) -> float:
        """one optimizer step with stratified minibatch.

        samples bs/2 from each group, concatenates, computes loss via loss_spec, backs up.
        returns loss as python float.
        """
        self.train()

        # effective batch size
        max_n = max(self.n_num, self.n_den)
        bs_eff = self.batch_size if (self.batch_size and self.batch_size < max_n) else max_n
        bs_half = bs_eff // 2

        # stratified sampling (with replacement)
        idx_num = torch.randint(0, self.n_num, (bs_half,), device=self.device)  # [bs_half]
        idx_den = torch.randint(0, self.n_den, (bs_half,), device=self.device)  # [bs_half]

        xb_num = self.samples_num[idx_num]  # [bs_half, input_dim]
        xb_den = self.samples_den[idx_den]  # [bs_half, input_dim]

        # concatenate: numerator first (label=1), then denominator (label=0)
        xb = torch.cat([xb_num, xb_den], dim=0)  # [bs_eff, input_dim]

        # build labels: 1 for numerator, 0 for denominator
        labels = torch.zeros(bs_eff, device=self.device)  # [bs_eff]
        labels[:bs_half] = 1  # numerator = class 1

        # forward + loss
        logits = self.forward(xb).squeeze(-1)  # [bs_eff, 1] -> [bs_eff]
        loss_scalar = self.loss_spec.loss_fn(logits, labels)  # scalar tensor

        # optimizer step
        self.optimizer.zero_grad()
        loss_scalar.backward()
        self.optimizer.step()

        return loss_scalar.item()

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        """inference: return model logits in eval mode with no_grad.

        xs: [N, input_dim]
        returns: [N]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(xs).squeeze(-1)  # [N, 1] -> [N]
        return logits

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """predict log density ratio via loss_spec.predict_ldr.

        xs: [N, input_dim]
        returns: [N]
        """
        logits = self.predict_logits(xs)  # [N]
        return self.loss_spec.predict_ldr(logits)  # [N]

    def finalize(self) -> None:
        """finalize: compute any loss_spec constants (e.g., c for DV) using denominator group."""
        logits_den = self.predict_logits(self.samples_den)  # [n_den]
        self.loss_spec.finalize(logits_den)


class StepMultiHeadCritic(nn.Module):
    """multi-head density ratio critic with independent per-head losses and stratified sampling.

    architecture:
        backbone (shared): input_dim -> latent_dim (ReLU) -> [latent_dim (ReLU) x (n_hidden_layers-1)]
        per-head (K independent): latent_dim -> latent_dim (ReLU) -> 1

    total backbone layers: n_hidden_layers.
    each head: 2 layers (latent->latent ReLU, latent->1 Linear).

    for each head k where loss_specs[k] is NWJSpec/DVSpec, final layer (latent->1) is zero-initialized.

    train_step sums per-head losses; predict_ldr returns sum of per-head log-density-ratios
    (telescoping identity holds after per-head constant subtraction via loss_spec.predict_ldr).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_hidden_layers: int,
        num_heads: int,
        loss_specs: list[LossSpec],
        lr: float,
        weight_decay: float,
        n_steps: int,
        batch_size: int | None,
        device: torch.device,
    ):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")
        if len(loss_specs) != num_heads:
            raise ValueError(f"loss_specs length {len(loss_specs)} != num_heads {num_heads}")

        # build shared backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(input_dim, latent_dim))
        backbone_layers.append(nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            backbone_layers.append(nn.Linear(latent_dim, latent_dim))
            backbone_layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone_layers)

        # build per-head parameters: latent_dim -> latent_dim (ReLU) -> 1
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        # first head layer: latent_dim -> latent_dim
        head_weight_0 = nn.Parameter(torch.empty(num_heads, latent_dim, latent_dim))  # [K, latent_dim, latent_dim]
        head_bias_0 = nn.Parameter(torch.empty(num_heads, latent_dim))  # [K, latent_dim]

        # final head layer: latent_dim -> 1
        head_weight_1 = nn.Parameter(torch.empty(num_heads, latent_dim, 1))  # [K, latent_dim, 1]
        head_bias_1 = nn.Parameter(torch.empty(num_heads, 1))  # [K, 1]

        self.head_weights = nn.ParameterList([head_weight_0, head_weight_1])
        self.head_biases = nn.ParameterList([head_bias_0, head_bias_1])

        self.to(device)
        self.device = device
        self.loss_specs = loss_specs
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size

        self._reset_parameters(loss_specs)

    def _reset_parameters(self, loss_specs: list[LossSpec]) -> None:
        """initialize backbone (xavier weights, zero biases), per-head layers.

        zero-init final layer per head for NWJSpec/DVSpec.
        """
        # backbone: xavier
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # head weights (all): xavier
        for W in self.head_weights:
            nn.init.xavier_uniform_(W.view(self.num_heads, -1))

        # head biases (all): zeros
        for b in self.head_biases:
            nn.init.zeros_(b)

        # zero-init final layer (head_weights[-1], head_biases[-1]) for NWJ/DV heads
        final_weight = self.head_weights[-1]  # [K, latent_dim, 1]
        final_bias = self.head_biases[-1]  # [K, 1]
        for k, loss_spec in enumerate(loss_specs):
            spec_name = loss_spec.__class__.__name__
            if spec_name in ("NWJSpec", "DVSpec"):
                nn.init.zeros_(final_weight[k])
                nn.init.zeros_(final_bias[k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with parallel head computation.

        x: [batch, input_dim]
        returns: [batch, num_heads]
        """
        features = self.backbone(x)  # [batch, latent_dim]

        # apply heads in parallel
        # first layer: [batch, latent_dim] x [K, latent_dim, latent_dim] -> [batch, K, latent_dim]
        x_heads = torch.einsum('bd,kdl->bkl', features, self.head_weights[0]) + self.head_biases[0]  # [batch, K, latent_dim]
        x_heads = torch.relu(x_heads)  # [batch, K, latent_dim]

        # final layer: [batch, K, latent_dim] x [K, latent_dim, 1] -> [batch, K, 1]
        x_heads = torch.einsum('bkd,kdl->bkl', x_heads, self.head_weights[1]) + self.head_biases[1]  # [batch, K, 1]

        return x_heads.squeeze(-1)  # [batch, K]

    def init_fit(
        self,
        samples_num_list: list[torch.Tensor],  # list of [M_k, input_dim], numerator per head
        samples_den_list: list[torch.Tensor],  # list of [M_k, input_dim], denominator per head
    ) -> None:
        """store sample lists and build optimizer.

        each head k has its own (possibly different-sized) sample groups.
        """
        self.samples_num_list = samples_num_list
        self.samples_den_list = samples_den_list
        self.n_num_list = [x.shape[0] for x in samples_num_list]  # [n_num_k, ...]
        self.n_den_list = [x.shape[0] for x in samples_den_list]  # [n_den_k, ...]
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_step(self) -> float:
        """one optimizer step with per-head stratified minibatch and summed losses.

        concatenates all head data, computes backbone once, splits, applies heads.
        returns total loss as python float.
        """
        self.train()

        # effective batch size (shared across heads)
        max_n = max(max(self.n_num_list), max(self.n_den_list))
        bs_eff = self.batch_size if (self.batch_size and self.batch_size < max_n) else max_n
        bs_half = bs_eff // 2

        # per-head sampling and data preparation
        xb_list = []
        labels_list = []

        for k in range(self.num_heads):
            # stratified sampling per head
            idx_num_k = torch.randint(0, self.n_num_list[k], (bs_half,), device=self.device)  # [bs_half]
            idx_den_k = torch.randint(0, self.n_den_list[k], (bs_half,), device=self.device)  # [bs_half]

            xb_num_k = self.samples_num_list[k][idx_num_k]  # [bs_half, input_dim]
            xb_den_k = self.samples_den_list[k][idx_den_k]  # [bs_half, input_dim]
            xb_k = torch.cat([xb_num_k, xb_den_k], dim=0)  # [bs_eff, input_dim]

            labels_k = torch.zeros(bs_eff, device=self.device)  # [bs_eff]
            labels_k[:bs_half] = 1  # numerator = class 1

            xb_list.append(xb_k)
            labels_list.append(labels_k)

        # concatenate all data; backbone pass once
        x_all = torch.cat(xb_list, dim=0)  # [K * bs_eff, input_dim]
        features_all = self.backbone(x_all)  # [K * bs_eff, latent_dim]

        # split features back per head
        features_list = torch.split(features_all, bs_eff, dim=0)  # K tensors of [bs_eff, latent_dim]

        # apply heads and accumulate losses
        total_loss = torch.tensor(0.0, device=self.device)
        for k in range(self.num_heads):
            features_k = features_list[k]  # [bs_eff, latent_dim]

            # apply head k: latent_dim -> latent_dim (ReLU) -> 1
            x_head_k = features_k @ self.head_weights[0][k] + self.head_biases[0][k]  # [bs_eff, latent_dim]
            x_head_k = torch.relu(x_head_k)  # [bs_eff, latent_dim]
            logits_k = (x_head_k @ self.head_weights[1][k] + self.head_biases[1][k]).squeeze(-1)  # [bs_eff]

            loss_k = self.loss_specs[k].loss_fn(logits_k, labels_list[k])  # scalar
            total_loss = total_loss + loss_k

        # optimizer step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def predict_logits(self, xs: torch.Tensor) -> torch.Tensor:
        """inference: return model logits per head in eval mode with no_grad.

        xs: [N, input_dim]
        returns: [N, num_heads]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(xs)  # [N, K]
        return logits

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """predict log density ratio as sum of per-head predictions.

        each head k computes loss_specs[k].predict_ldr(logits[:, k]), then sum.
        (per-head constant already subtracted, telescoping identity holds.)

        xs: [N, input_dim]
        returns: [N]
        """
        logits = self.predict_logits(xs)  # [N, K]
        ldr = torch.zeros(xs.shape[0], device=xs.device)  # [N]
        for k in range(self.num_heads):
            logits_k = logits[:, k]  # [N]
            ldr = ldr + self.loss_specs[k].predict_ldr(logits_k)  # [N]
        return ldr

    def finalize(self) -> None:
        """per-head finalization: compute constants (e.g., c for DV) using denominator groups.

        uses eval mode and no_grad for stable constant computation.
        """
        self.eval()
        with torch.no_grad():
            for k in range(self.num_heads):
                features_den_k = self.backbone(self.samples_den_list[k])  # [n_den_k, latent_dim]

                # apply head k to denominator features
                x_head_k = features_den_k @ self.head_weights[0][k] + self.head_biases[0][k]  # [n_den_k, latent_dim]
                x_head_k = torch.relu(x_head_k)  # [n_den_k, latent_dim]
                logits_den_k = (x_head_k @ self.head_weights[1][k] + self.head_biases[1][k]).squeeze(-1)  # [n_den_k]

                self.loss_specs[k].finalize(logits_den_k)
