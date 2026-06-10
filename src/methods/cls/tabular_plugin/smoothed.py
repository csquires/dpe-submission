"""SmoothedTabularPluginDRE: oracle smoothed DRE using ground-truth latent indices."""
import torch

from ...common.base import DRE
from src.sampling.tabular import pointwise_smoothed_ldr
from ._common import count_and_smooth


class SmoothedTabularPluginDRE(DRE):
    """oracle density ratio estimator using ground-truth latent indices.

    evaluates closed-form smoothed log-density-ratios via the encoding kernel.
    only supports continuous smoothed encodings: gaussian_blob, flow_pushforward.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        encoding_cfg: dict,
        smoothing_alpha: float = 0.5,
        device: str = "cuda",
        early_stop_cfg: dict | None = None,
    ):
        """initialize oracle estimator and validate encoding."""
        enc_type = encoding_cfg["type"]
        input_dim = encoding_cfg["embed_dim"]

        super().__init__(input_dim)

        self.n_states = n_states
        self.n_actions = n_actions
        self.encoding_cfg = encoding_cfg
        self.smoothing_alpha = smoothing_alpha
        self.device = device
        self.early_stop_cfg = early_stop_cfg
        self._d_O_hat = None
        self._d_E_hat = None
        self._fitted = False

        assert enc_type in {"gaussian_blob", "flow_pushforward"}, \
            "SmoothedTabularPluginDRE only supports gaussian_blob or flow_pushforward encodings"

        assert smoothing_alpha > 0, "smoothing_alpha must be > 0"

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        latent_p0: torch.Tensor = None,
        latent_p1: torch.Tensor = None,
    ) -> None:
        """build empirical occupancy distributions from ground-truth latent indices."""
        assert not self._fitted, "fit already called"

        assert latent_p0 is not None, "SmoothedTabularPluginDRE requires latent_p0 at fit time"
        assert latent_p1 is not None, "SmoothedTabularPluginDRE requires latent_p1 at fit time"
        assert latent_p0.shape[1] == 2 and latent_p0.dtype == torch.int64
        assert latent_p1.shape[1] == 2 and latent_p1.dtype == torch.int64

        s_O = latent_p0[:, 0]
        a_O = latent_p0[:, 1]
        s_E = latent_p1[:, 0]
        a_E = latent_p1[:, 1]

        self._d_O_hat = count_and_smooth(s_O, a_O, self.n_states, self.n_actions, self.smoothing_alpha).cpu()
        self._d_E_hat = count_and_smooth(s_E, a_E, self.n_states, self.n_actions, self.smoothing_alpha).cpu()

        self._final_step = 0
        self._stop_reason = None
        self._fitted = True

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """evaluate closed-form smoothed log-density-ratios via the encoding kernel."""
        assert self._fitted, "fit must be called before predict_ldr"

        ldrs = pointwise_smoothed_ldr(
            xs,
            self.encoding_cfg,
            self._d_O_hat.numpy(),
            self._d_E_hat.numpy()
        )

        if not isinstance(ldrs, torch.Tensor):
            ldrs = torch.from_numpy(ldrs).float()

        ldrs = ldrs.to(xs.device)
        return ldrs
