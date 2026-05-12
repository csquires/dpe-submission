"""Phase 1 pilot config: BDRE on dre_sample_complexity.

Validates Optuna integration end-to-end before rolling out further.
"""

from experiments.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="dre_sample_complexity",
    methods=["BDRE"],
    min_resource=100,
    max_resource=10000,
    reduction_factor=3,
    holdout_top_k=5,
    walltime_minutes=120,
    walltime_margin_minutes=10,
    resume_existing=True,
    include_tabular=False,
)
