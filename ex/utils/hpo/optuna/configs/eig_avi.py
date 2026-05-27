"""EIGAdapter HPO config — aviamala campaign.

methods on aviamala: CTSM family + TSM. cls methods (BDRE/MDRE/TriangularMDRE/
MultiHeadTriangularTDRE) and TriangularTSM deferred to a separate campaign
once their respective fit-pipeline issues are resolved. bvarici runs the VFM
and FMDRE families in parallel.
"""

from ex.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="eig",
    methods=[
        # score-matching reg lines (src.methods.reg).
        "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
        "TSM",
    ],
    target_trials=512,
    walltime_minutes=120,
    min_resource=400,
    max_resource=6400,
    reduction_factor=2,
    holdout_top_k=5,
    walltime_margin_minutes=10,
    fixed_hp={"n_hidden_layers": 5},
    lanes=["general", "preempt", "cpu", "array"],
    resume_existing=True,
    include_tabular=False,
)
