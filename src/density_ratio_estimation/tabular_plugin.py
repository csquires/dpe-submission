import torch
import torch.nn.functional as F
from src.density_ratio_estimation.base import DensityRatioEstimator
from src.sampling.tabular import grid_coord_angle, pointwise_smoothed_ldr


def _count_and_smooth(s: torch.Tensor, a: torch.Tensor, n_states: int, n_actions: int, alpha: float) -> torch.Tensor:
	"""count (s, a) pairs and apply laplace smoothing.

	flattens (s, a) to joint index k = s * n_actions + a, histograms via bincount,
	adds laplace smoothing constant alpha, normalizes to density, and reshapes to
	tabular form [n_states, n_actions].

	args:
		s: state indices, shape [N], int64
		a: action indices, shape [N], int64
		n_states: number of states
		n_actions: number of actions
		alpha: laplace smoothing parameter (> 0)

	returns:
		d_hat: empirical density [n_states, n_actions], sums to 1.0
	"""
	K = n_states * n_actions
	flat = s * n_actions + a  # [N]
	count = torch.bincount(flat, minlength=K).float() + alpha  # [K]
	d = count / count.sum()  # [K]
	return d.reshape(n_states, n_actions)


class TabularPluginDRE(DensityRatioEstimator):
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
        """
        # derive input_dim from encoding type
        enc_type = encoding_cfg["type"]
        if enc_type == "onehot_joint":
            input_dim = n_states * n_actions
        elif enc_type == "onehot_concat":
            input_dim = n_states + n_actions
        else:  # gaussian_blob or flow_pushforward
            input_dim = encoding_cfg["embed_dim"]

        super().__init__(input_dim)

        self.n_states = n_states
        self.n_actions = n_actions
        self.encoding_cfg = encoding_cfg
        self.decode = decode
        self.smoothing_alpha = smoothing_alpha
        self.device = device
        self._d_O_hat = None
        self._d_E_hat = None
        self._fitted = False

        # validate smoothing_alpha
        assert smoothing_alpha > 0, "smoothing_alpha must be > 0"

        # validate encoding-decode pairing
        if enc_type in {"onehot_joint", "onehot_concat"}:
            if decode != "argmax":
                raise ValueError("onehot encodings must use decode='argmax'")
        elif enc_type in {"gaussian_blob", "flow_pushforward"}:
            if decode not in {"argmax", "nn"}:
                raise ValueError("blob/flow encodings must use decode in {'argmax', 'nn'}")

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """estimate empirical marginal distributions via count-based statistics.

        args:
            samples_p0: encoded samples from p0, shape [N_O, input_dim]
            samples_p1: encoded samples from p1, shape [N_E, input_dim]
        """
        assert not self._fitted, "fit already called"

        # decode samples to discrete (s, a) indices
        s_O, a_O = self._decode(samples_p0)  # each shape [N_O], int64
        s_E, a_E = self._decode(samples_p1)  # each shape [N_E], int64

        # count and smooth via helper
        self._d_O_hat = _count_and_smooth(s_O, a_O, self.n_states, self.n_actions, self.smoothing_alpha).cpu()  # [n_states, n_actions]
        self._d_E_hat = _count_and_smooth(s_E, a_E, self.n_states, self.n_actions, self.smoothing_alpha).cpu()  # [n_states, n_actions]

        self._fitted = True

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """compute log-density-ratio at query points.

        args:
            xs: encoded query points, shape [N, input_dim]

        returns:
            ldr: log-density-ratios, shape [N]
        """
        assert self._fitted, "fit must be called before predict_ldr"

        # decode to discrete (s, a) indices
        s, a = self._decode(xs)  # each shape [N], int64

        # move distributions to xs device for indexing
        d_O_hat = self._d_O_hat.to(xs.device)  # [n_states, n_actions]
        d_E_hat = self._d_E_hat.to(xs.device)  # [n_states, n_actions]

        # index and compute log-ratio
        log_d_O = torch.log(d_O_hat[s, a])  # [N]
        log_d_E = torch.log(d_E_hat[s, a])  # [N]
        ldr = log_d_O - log_d_E  # [N]

        return ldr

    def _decode(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """convert encoded representations to discrete (s, a) indices.

        args:
            xs: encoded samples, shape [N, input_dim]

        returns:
            (s, a): state and action indices, each shape [N], int64

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
            # decode one-hot joint encoding
            joint = xs.argmax(dim=-1)  # [N]
            s = joint // self.n_actions
            a = joint % self.n_actions
            return s, a

        elif enc_type == "onehot_concat":
            # decode concatenated one-hot encodings
            s_one_hot = xs[:, :self.n_states]  # [N, n_states]
            a_one_hot = xs[:, self.n_states:]  # [N, n_actions]
            s = s_one_hot.argmax(dim=-1)  # [N]
            a = a_one_hot.argmax(dim=-1)  # [N]
            return s, a

        elif enc_type == "gaussian_blob":
            if self.decode == "argmax":
                raise ValueError("gaussian_blob with decode='argmax' is invalid; use decode='nn'")
            elif self.decode == "nn":
                # nearest neighbor decoding in embedding space
                n_states = self.n_states
                n_actions = self.n_actions
                K = n_states * n_actions

                # build grid of all (s, a) embeddings via meshgrid + grid_coord_angle
                s_grid, a_grid = torch.meshgrid(
                    torch.arange(n_states, device=xs.device),
                    torch.arange(n_actions, device=xs.device),
                    indexing="ij"
                )
                s_grid = s_grid.reshape(-1)  # [K]
                a_grid = a_grid.reshape(-1)  # [K]
                phi_grid = grid_coord_angle(
                    s_grid, a_grid, self.encoding_cfg["L"], n_actions
                )  # [K, embed_dim]

                # squared euclidean distance: [N, K]
                dist = (xs[:, None, :] - phi_grid[None, :, :]).pow(2).sum(dim=-1)

                # nearest neighbor
                flat = dist.argmin(dim=-1)  # [N]
                s = flat // n_actions
                a = flat % n_actions
                return s, a

        elif enc_type == "flow_pushforward":
            if self.decode == "argmax":
                raise ValueError("flow_pushforward with decode='argmax' is invalid; use decode='nn'")
            elif self.decode == "nn":
                # inverse flow to latent space, then nearest neighbor
                flow = self.encoding_cfg["flow_module"]
                z, _ = flow.inverse(xs)  # [N, embed_dim]

                n_states = self.n_states
                n_actions = self.n_actions
                K = n_states * n_actions

                # build grid in latent space
                s_grid, a_grid = torch.meshgrid(
                    torch.arange(n_states, device=z.device),
                    torch.arange(n_actions, device=z.device),
                    indexing="ij"
                )
                s_grid = s_grid.reshape(-1)  # [K]
                a_grid = a_grid.reshape(-1)  # [K]
                phi_grid = grid_coord_angle(
                    s_grid, a_grid, self.encoding_cfg["L"], n_actions
                )  # [K, embed_dim]

                # squared euclidean distance in latent space: [N, K]
                dist = (z[:, None, :] - phi_grid[None, :, :]).pow(2).sum(dim=-1)

                # nearest neighbor
                flat = dist.argmin(dim=-1)  # [N]
                s = flat // n_actions
                a = flat % n_actions
                return s, a


class SmoothedTabularPluginDRE(DensityRatioEstimator):
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
    ):
        """initialize oracle estimator and validate encoding.

        args:
            n_states: number of states
            n_actions: number of actions
            encoding_cfg: encoding configuration dict. must have type in {gaussian_blob, flow_pushforward}
            smoothing_alpha: laplace smoothing parameter (> 0)
            device: torch device
        """
        # derive input_dim from encoding type
        enc_type = encoding_cfg["type"]
        input_dim = encoding_cfg["embed_dim"]

        super().__init__(input_dim)

        self.n_states = n_states
        self.n_actions = n_actions
        self.encoding_cfg = encoding_cfg
        self.smoothing_alpha = smoothing_alpha
        self.device = device
        self._d_O_hat = None
        self._d_E_hat = None
        self._fitted = False

        # validate encoding type
        assert enc_type in {"gaussian_blob", "flow_pushforward"}, \
            "SmoothedTabularPluginDRE only supports gaussian_blob or flow_pushforward encodings"

        # validate smoothing_alpha
        assert smoothing_alpha > 0, "smoothing_alpha must be > 0"

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        latent_p0: torch.Tensor = None,
        latent_p1: torch.Tensor = None,
    ) -> None:
        """build empirical occupancy distributions from ground-truth latent indices.

        args:
            samples_p0: encoded samples from p0 (unused; kept for interface compatibility)
            samples_p1: encoded samples from p1 (unused; kept for interface compatibility)
            latent_p0: ground-truth (s, a) indices for p0, shape [N_O, 2], int64
            latent_p1: ground-truth (s, a) indices for p1, shape [N_E, 2], int64
        """
        assert not self._fitted, "fit already called"

        # validate oracle inputs
        assert latent_p0 is not None, "SmoothedTabularPluginDRE requires latent_p0 at fit time"
        assert latent_p1 is not None, "SmoothedTabularPluginDRE requires latent_p1 at fit time"
        assert latent_p0.shape[1] == 2 and latent_p0.dtype == torch.int64
        assert latent_p1.shape[1] == 2 and latent_p1.dtype == torch.int64

        # extract (s, a) pairs
        s_O = latent_p0[:, 0]  # [N_O]
        a_O = latent_p0[:, 1]  # [N_O]
        s_E = latent_p1[:, 0]  # [N_E]
        a_E = latent_p1[:, 1]  # [N_E]

        # count and smooth via helper
        self._d_O_hat = _count_and_smooth(s_O, a_O, self.n_states, self.n_actions, self.smoothing_alpha).cpu()  # [n_states, n_actions]
        self._d_E_hat = _count_and_smooth(s_E, a_E, self.n_states, self.n_actions, self.smoothing_alpha).cpu()  # [n_states, n_actions]

        self._fitted = True

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """evaluate closed-form smoothed log-density-ratios via the encoding kernel.

        args:
            xs: encoded query points, shape [N, embed_dim]

        returns:
            ldr: log-density-ratios, shape [N]
        """
        assert self._fitted, "fit must be called before predict_ldr"

        # call library function for smoothed kernel-based evaluation
        ldrs = pointwise_smoothed_ldr(
            xs,
            self.encoding_cfg,
            self._d_O_hat.numpy(),
            self._d_E_hat.numpy()
        )

        # ensure tensor and device compatibility
        if not isinstance(ldrs, torch.Tensor):
            ldrs = torch.from_numpy(ldrs).float()

        ldrs = ldrs.to(xs.device)
        return ldrs
