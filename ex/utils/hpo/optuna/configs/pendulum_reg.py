"""HPO config: pendulum experiment with VFM-family methods."""

from ex.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="pendulum",
    methods=["VFM", "VFMOrthros", "MultiHeadTriangularTDRE", "MultiHeadTDRE"],
    min_resource=100,
    max_resource=1600,
    reduction_factor=2,
    holdout_top_k=5,
    walltime_minutes=120,
    walltime_margin_minutes=10,
    fixed_hp={"n_hidden_layers": 4},
    lanes=["general"],
    resume_existing=True,
    include_tabular=False,
)
