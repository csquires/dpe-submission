"""HPO config: pendulum experiment with VFM-family methods."""

from ex.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="pendulum",
    methods=[
        # cls-based (src.methods.cls). triangular re-done with nwp={3,5,7,9};
        # BDRE/MDRE fresh (weight_decay-ctor bug fixed). non-triangular
        # MultiHeadTDRE FROZEN at its done state -- excluded from this campaign.
        "MultiHeadTriangularTDRE",
        "BDRE", "MDRE", "TriangularMDRE",
        # reg-based (src.methods.reg) -- fresh studies, new search space, n_steps=6400.
        "VFM", "VFMOrthros", "CTSM", "FMDRE", "FMDRE_S2", "TSM", "TriangularFMDRE",
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
