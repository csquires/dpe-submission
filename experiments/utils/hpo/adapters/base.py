"""unifies per-experiment data + h5 loading + metric semantics. cell tuples are arity-agnostic (1, 2, or 3); adapter encodes shape via cell_pool() return type."""
import abc
from pathlib import Path
from typing import Optional
import torch


class ExperimentAdapter(abc.ABC):
    """abstract interface for HPO trial experiment configuration.

    each subclass encapsulates one experiment's data loading, cell pool definition,
    and metric naming. all cell operations are tuple-agnostic; tuples are opaque
    int sequences. subclass docstring should declare cell arity for downstream
    validation. subclasses own h5 key mapping and file I/O.
    """

    @abc.abstractmethod
    def name(self) -> str:
        """short experiment identifier for seed namespace and yaml keys.

        e.g. "mnist_cond_flow", "pendulum", "eig". used by trial_runner as
        experiment= param.
        """

    @abc.abstractmethod
    def data_dir(self) -> Path:
        """NFS-shared data root directory.

        subclass may append exp-specific subdirs in load_cell_data().
        """

    @abc.abstractmethod
    def cell_pool(self) -> list[tuple[int, ...]]:
        """full evaluation cell list; opaque tuples of ints.

        arity varies per experiment (declared in subclass docstring).
        e.g. mnist returns [(0,0), ..., (3,39)]; pendulum returns
        [(k1, k2, seed), ...]; model_selection returns [(row_idx,), ...].
        """

    @abc.abstractmethod
    def load_cell_data(self, cell: tuple[int, ...], device: str) -> dict[str, torch.Tensor]:
        """load one cell from H5; return normalized dict.

        required keys: 'pstar', 'p0', 'p1', 'true_ldrs' (all tensors on device).
        h5 key names and file paths are exp-specific; subclass owns the mapping.
        raises FileNotFoundError or ValueError if cell invalid or file missing.
        """

    @abc.abstractmethod
    def device(self) -> str:
        """torch device string, e.g. 'cpu' or 'cuda'.

        typically read from experiment's config.yaml.
        """

    @abc.abstractmethod
    def latent_dim(self) -> int:
        """input dimension for estimator builder.

        for mnist/dbpedia: 14; for pendulum: 4; for eig: data_dim + 1;
        for smodice: determined by encoding type.
        """

    @abc.abstractmethod
    def num_waypoints(self) -> Optional[int]:
        """number of waypoints for triangular methods, or None if not used.

        if None, triangular methods adapt by setting pstar = p0 as fallback
        (eig pattern). if int, passed to builder.
        """

    @abc.abstractmethod
    def metric_key(self) -> str:
        """metric dict key in result JSON returned by run_trial().

        examples: "per_pair_mae" (mnist/dbpedia), "per_cell_ldr_mae"
        (pendulum/smodice), "per_design_eig_abs_err" (eig),
        "per_cell_eldr_abs_err" (elbo), "per_row_ldr_mean_ae" (model_selection).
        """

    def cell_seed_namespace(self) -> str:
        """optional seed namespace prefix for run_trial().

        default: name(). override if experiment uses a custom seed prefix.
        """
        return self.name()

    def is_ready(self) -> bool:
        """check if data_dir exists on NFS.

        default: True if path exists. subclass may override to validate config
        gates (e.g., pendulum gates on kl_targets.k1_values populated).
        """
        return self.data_dir().exists()

    def supports_tabular(self) -> bool:
        """flag for tabular-only methods (e.g., model_selection, smodice).

        default: False. only smodice overrides to True; launcher filters
        methods by this flag.
        """
        return False

    def default_training_M(self) -> int:
        """default number of training cells sampled by cell_schema.

        default: 32 (matches v735e random_b convention). override e.g.
        model_selection → 5 if pool is small.
        """
        return 32

    def default_holdout_M(self) -> int:
        """default number of holdout cells sampled by cell_schema after
        excluding training cells.

        default: 32. caller may pick smaller holdout if (pool - training) < default;
        cell_schema clamps with logged warning.
        """
        return 32

    def eval_cell(
        self,
        cell: tuple[int, ...],
        method: str,
        builder,
        hyperparams: dict,
        requires_pstar: bool,
        device: str,
        *,
        data: Optional[dict] = None,
    ) -> float:
        """build estimator, fit on this cell, return scalar metric.

        default impl assumes the {p0, p1, pstar, true_ldrs} schema used by
        mnist/dbpedia/pendulum/smodice/dre_sample_complexity adapters and
        scores via mae(predict_ldr(pstar), true_ldrs). adapters with a
        different data schema or metric (eig, elbo, model_selection) MUST
        override this method.

        if data is provided (cpu_runner cache path), reuse it; else load
        from h5 via load_cell_data.

        plan:
          1. data = data or load_cell_data(cell, device).
          2. build estimator: pop num_waypoints from hp (HP-driven if present,
             else adapter default), pass remaining hp via **flat.
          3. fit on (p0, p1[, pstar]) per requires_pstar.
          4. predict_ldr(pstar) -> mae vs true_ldrs.

        in-flight mnist runs: unchanged (this default reproduces the
        pre-existing inline logic from trial_runner / cpu_runner).
        """
        if data is None:
            data = self.load_cell_data(cell, device=device)
        nwp = hyperparams.get("num_waypoints", self.num_waypoints())
        flat = {k: v for k, v in hyperparams.items() if k != "num_waypoints"}
        est = builder(
            input_dim=self.latent_dim(),
            device=device,
            num_waypoints=nwp,
            **flat,
        )
        if requires_pstar:
            est.fit(data["p0"], data["p1"], data["pstar"])
        else:
            est.fit(data["p0"], data["p1"])
        with torch.no_grad():
            predicted = est.predict_ldr(data["pstar"])
            return float(torch.abs(predicted.cpu() - data["true_ldrs"].cpu()).mean())

    def stratify_key(self, cell: tuple[int, ...]):
        """optional stratification key for the cell. if any adapter cell returns
        non-None, cell_schema.draw_training_sample switches to stratified mode:
        groups cells by this key and samples K_per_stratum = max(1, M//n_groups)
        from each group.

        default: None (un-stratified, current behavior).

        override examples:
          smodice / pendulum: return (k1, k2) — guarantees every (k1, k2) regime
            is sampled, avoiding the ~47%-miss-rate of naive random sampling
            from a 480-cell pool.
          elbo: return alpha — guarantees per-alpha coverage.
          mnist / dbpedia: keep None — alpha coverage is implicit by random draw
            from 160-cell pool.
        """
        return None
