"""HPO config: occupancy gridworld -- gold-standard reproducer.

single tracked config covering every method we report on occupancy. running
this from an empty journal reproduces our full occupancy sweep end-to-end.
account- or campaign-specific subsets live in untracked sibling files
(e.g. occupancy_fillin.py).

methods are grouped by family:
  - cls-based (src.methods.cls): BDRE, MDRE, MultiHeadTDRE + triangular variants.
  - TSM-line (src.methods.reg):  TSM, TriangularTSM.
  - CTSM-line:                   CTSM, TriangularCTSM V1-V3.
  - VFM-line:                    VFM, VFMOrthros, TriangularVFM V1-V3.
  - FMDRE-line:                  FMDRE, FMDRE_S2, TriangularFMDRE.

tabular plugin DRE is excluded (include_tabular=False).
"""

from ex.utils.hpo.optuna.study_config import StudyConfig


CONFIG = StudyConfig(
    study_seed=1729,
    experiment="occupancy",
    methods=[
        "BDRE", "MDRE", "MultiHeadTDRE",
        "TriangularMDRE", "MultiHeadTriangularTDRE",
        "TSM", "TriangularTSM",
        "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
        "VFM", "VFMOrthros",
        "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3",
        "FMDRE", "FMDRE_S2", "TriangularFMDRE",
    ],
    target_trials=512,
    walltime_minutes=120,
    walltime_margin_minutes=10,
    min_resource=400,
    max_resource=6400,
    reduction_factor=2,
    holdout_top_k=5,
    fixed_hp={"n_hidden_layers": 5},
    lanes=["general", "preempt", "cpu", "array"],
    resume_existing=True,
    include_tabular=False,
)
