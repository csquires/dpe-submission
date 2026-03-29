import torch

from src.eig_estimation.base import EIGEstimation
from src.density_ratio_estimation.base import DensityRatioEstimator


class EIGPlugin(EIGEstimation):
    def __init__(
        self,
        density_ratio_estimator: DensityRatioEstimator,
        train_ratio: float = None,
    ):
        """
        args:
            density_ratio_estimator: DRE model
            train_ratio: if set, use this fraction for training and rest for eval.
                         if None, use all samples for both (original behavior).
        """
        self.density_ratio_estimator = density_ratio_estimator
        self.train_ratio = train_ratio

    def _create_marginal_samples(self, samples_theta: torch.Tensor, samples_y: torch.Tensor) -> torch.Tensor:
        shuffled_thetas = samples_theta[torch.randperm(samples_theta.shape[0])]
        shuffled_ys = samples_y[torch.randperm(samples_y.shape[0])]
        return torch.cat([shuffled_thetas, shuffled_ys], dim=1)

    def estimate_eig(
        self,
        samples_theta: torch.Tensor,
        samples_y: torch.Tensor,
    ) -> float:
        n = samples_theta.shape[0]

        if self.train_ratio is not None:
            # split into train and eval
            n_train = int(n * self.train_ratio)
            perm = torch.randperm(n, device=samples_theta.device)
            train_idx, eval_idx = perm[:n_train], perm[n_train:]

            train_theta, eval_theta = samples_theta[train_idx], samples_theta[eval_idx]
            train_y, eval_y = samples_y[train_idx], samples_y[eval_idx]

            # create train samples
            train_p0 = torch.cat([train_theta, train_y], dim=1)
            train_p1 = self._create_marginal_samples(train_theta, train_y)

            # create eval samples (joint only - we estimate E_p0[log r])
            eval_p0 = torch.cat([eval_theta, eval_y], dim=1)
        else:
            # original behavior: use all for both
            train_p0 = torch.cat([samples_theta, samples_y], dim=1)
            train_p1 = self._create_marginal_samples(samples_theta, samples_y)
            eval_p0 = train_p0

        # fit and predict
        self.density_ratio_estimator.fit(train_p0, train_p1)
        est_ldrs = self.density_ratio_estimator.predict_ldr(eval_p0)
        return torch.mean(est_ldrs)


if __name__ == "__main__":
    from src.density_ratio_estimation import BDRE
    from src.models.binary_classification import make_binary_classifier

    DATA_DIM = 2

    # build estimator
    classifier = make_binary_classifier(name="default", input_dim=DATA_DIM+1)
    density_ratio_estimator = BDRE(classifier, device="cuda")
    eig_plugin = EIGPlugin(density_ratio_estimator=density_ratio_estimator)

    # generate fake data
    samples_theta = torch.randn(1000, DATA_DIM).to("cuda")
    samples_y = torch.randn(1000, 1).to("cuda")

    # estimate eig
    est_eig = eig_plugin.estimate_eig(samples_theta, samples_y)
    print(f"Estimated EIG: {est_eig}")