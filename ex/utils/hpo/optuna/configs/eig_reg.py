"""EIGAdapter HPO config.

GPU serialization is handled by the preempt lane (LaneProfile.batch_size=1 in
ex/utils/hpo/optuna/lanes.py). EIGAdapter HPO runs single-process per the lane,
and eig's adapter already performs runtime CUDA device discovery.
"""

from ex.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="eig",
    methods=[
        # cls-based (src.methods.cls). triangular re-done with nwp={3,5,7,9};
        # BDRE/MDRE fresh (weight_decay-ctor bug fixed). non-triangular
        # MultiHeadTDRE FROZEN at its done state -- excluded from this campaign.
        "MultiHeadTriangularTDRE",
        "BDRE", "MDRE", "TriangularMDRE",
        # reg-based (src.methods.reg) -- fresh studies, new search space, n_epochs=4000.
        "VFM", "VFMOrthros", "CTSM", "FMDRE", "FMDRE_S2", "TSM", "TriangularFMDRE",
    ],
    target_trials=512,
    walltime_minutes=120,
    min_resource=100,
    max_resource=3200,
    reduction_factor=2,
    holdout_top_k=5,
    walltime_margin_minutes=10,
    fixed_hp={"n_hidden_layers": 3},
    lanes=["general", "preempt", "cpu", "array"],
    resume_existing=True,
    include_tabular=False,
)
