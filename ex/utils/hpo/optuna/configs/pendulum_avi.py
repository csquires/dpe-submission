"""HPO config: pendulum experiment, aviamala campaign.

see eig_avi.py for the cls / CTSM-line / TSM-line split rationale.
"""

from ex.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="pendulum",
    methods=[
        "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
        "TSM",
    ],
    target_trials=512,
    min_resource=400,
    max_resource=6400,
    reduction_factor=2,
    holdout_top_k=5,
    walltime_minutes=120,
    walltime_margin_minutes=10,
    fixed_hp={"n_hidden_layers": 5},
    lanes=["general", "preempt", "cpu", "array"],
    resume_existing=True,
    include_tabular=False,
)
