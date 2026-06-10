"""TabularPluginDRE: empirical plug-in DRE for tabular (state, action) pairs."""
import torch

from ...common.base import DRE
from src.sampling.tabular import grid_coord_angle
from ._common import count_and_smooth


class TabularPluginDRE(DRE):
    """empirical plug-in density ratio estimator for tabular (state, action) pairs.

    supports four encoding types: onehot_joint, onehot_concat, gaussian_blob, flow_pushforward.
    decoding strategy depends on encoding type:
    - onehot_joint/onehot_concat: decode="argmax" (required).
    - gaussian_blob/flow_pushforward: decode in {"argmax", "nn"}.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        encoding_cfg: dict,
        decode: str = "argmax",
        smoothing_alpha: float = 0.5,
        device: str = "cuda",
        early_stop_cfg: dict | None = None,
    ):
        """initialize estimator and validate configuration.

        args:
            n_states: number of states
            n_actions: number of actions
            encoding_cfg: encoding configuration dict with keys:
                - "type": one of {onehot_joint, onehot_concat, gaussian_blob, flow_pushforward}
                - "embed_dim": embedding dimension (for gaussian_blob, flow_pushforward)
                - "L", "flow_module" as needed per type
            decode: decoding strategy. argmax for onehot, "argmax" or "nn" for blobs/flows
            smoothing_alpha: laplace smoothing parameter (> 0)
            device: torch device
            early_stop_cfg: early stopping configuration (unused for closed-form estimator)
        """
        enc_type = encoding_cfg["type"]
        if enc_type == "onehot_joint":
            input_dim = n_states * n_actions
        elif enc_type == "onehot_concat":
            input_dim = n_states + n_actions
        else:
            input_dim = encoding_cfg["embed_dim"]

        super().__init__(input_dim)

        self.n_states = n_states
        self.n_actions = n_actions
        self.encoding_cfg = encoding_cfg
        self.decode = decode
        self.smoothing_alpha = smoothing_alpha
        self.device = device
        self.early_stop_cfg = early_stop_cfg
        self._d_O_hat = None
        self._d_E_hat = None
        self._fitted = False

        assert smoothing_alpha > 0, "smoothing_alpha must be > 0"

        if enc_type in {"onehot_joint", "onehot_concat"}:
            if decode != "argmax":
                raise ValueError("onehot encodings must use decode='argmax'")
        elif enc_type in {"gaussian_blob", "flow_pushforward"}:
            if decode not in {"argmax", "nn"}:
                raise ValueError("blob/flow encodings must use decode in {'argmax', 'nn'}")

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """estimate empirical marginal distributions via count-based statistics."""
        assert not self._fitted, "fit already called"

        s_O, a_O = self._decode(samples_p0)
        s_E, a_E = self._decode(samples_p1)

        self._d_O_hat = count_and_smooth(s_O, a_O, self.n_states, self.n_actions, self.smoothing_alpha).cpu()
        self._d_E_hat = count_and_smooth(s_E, a_E, self.n_states, self.n_actions, self.smoothing_alpha).cpu()

        self._fitted = True
        self._final_step = 0
        self._stop_reason = None

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """compute log-density-ratio at query points."""
        assert self._fitted, "fit must be called before predict_ldr"

        s, a = self._decode(xs)

        d_O_hat = self._d_O_hat.to(xs.device)
        d_E_hat = self._d_E_hat.to(xs.device)

        log_d_O = torch.log(d_O_hat[s, a])
        log_d_E = torch.log(d_E_hat[s, a])
        return log_d_O - log_d_E

    def _decode(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """convert encoded representations to discrete (s, a) indices.

        dispatch table:
        - onehot_joint + argmax: joint.argmax -> s, a
        - onehot_concat + argmax: split and argmax separately
        - gaussian_blob + argmax: error
        - gaussian_blob + nn: nearest neighbor in embedding space
        - flow_pushforward + argmax: error
        - flow_pushforward + nn: inverse flow, nearest neighbor in latent space
        """
        enc_type = self.encoding_cfg["type"]

        if enc_type == "onehot_joint":
            joint = xs.argmax(dim=-1)
            s = joint // self.n_actions
            a = joint % self.n_actions
            return s, a

        elif enc_type == "onehot_concat":
            s_one_hot = xs[:, :self.n_states]
            a_one_hot = xs[:, self.n_states:]
            s = s_one_hot.argmax(dim=-1)
            a = a_one_hot.argmax(dim=-1)
            return s, a

        elif enc_type == "gaussian_blob":
            if self.decode == "argmax":
                raise ValueError("gaussian_blob with decode='argmax' is invalid; use decode='nn'")
            elif self.decode == "nn":
                n_states = self.n_states
                n_actions = self.n_actions

                s_grid, a_grid = torch.meshgrid(
                    torch.arange(n_states, device=xs.device),
                    torch.arange(n_actions, device=xs.device),
                    indexing="ij"
                )
                s_grid = s_grid.reshape(-1)
                a_grid = a_grid.reshape(-1)
                phi_grid = grid_coord_angle(s_grid, a_grid, self.encoding_cfg["L"], n_actions)

                dist = (xs[:, None, :] - phi_grid[None, :, :]).pow(2).sum(dim=-1)
                flat = dist.argmin(dim=-1)
                s = flat // n_actions
                a = flat % n_actions
                return s, a

        elif enc_type == "flow_pushforward":
            if self.decode == "argmax":
                raise ValueError("flow_pushforward with decode='argmax' is invalid; use decode='nn'")
            elif self.decode == "nn":
                flow = self.encoding_cfg["flow_module"]
                z, _ = flow.inverse(xs)

                n_states = self.n_states
                n_actions = self.n_actions

                s_grid, a_grid = torch.meshgrid(
                    torch.arange(n_states, device=z.device),
                    torch.arange(n_actions, device=z.device),
                    indexing="ij"
                )
                s_grid = s_grid.reshape(-1)
                a_grid = a_grid.reshape(-1)
                phi_grid = grid_coord_angle(s_grid, a_grid, self.encoding_cfg["L"], n_actions)

                dist = (z[:, None, :] - phi_grid[None, :, :]).pow(2).sum(dim=-1)
                flat = dist.argmin(dim=-1)
                s = flat // n_actions
                a = flat % n_actions
                return s, a
