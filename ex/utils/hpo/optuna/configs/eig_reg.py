"""EIGAdapter HPO config.

GPU serialization is handled by the preempt lane (LaneProfile.batch_size=1 in
ex/utils/hpo/optuna/lanes.py). EIGAdapter HPO runs single-process per the lane,
and eig's adapter already performs runtime CUDA device discovery.
"""

from ex.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="eig",
    methods=["VFM", "VFMOrthros", "MultiHeadTriangularTDRE", "MultiHeadTDRE"],
    walltime_minutes=120,
    min_resource=100,
    max_resource=1600,
    reduction_factor=2,
    holdout_top_k=5,
    walltime_margin_minutes=10,
    fixed_hp={"n_hidden_layers": 3},
    lanes=["cpu"],
    resume_existing=True,
    include_tabular=False,
)
