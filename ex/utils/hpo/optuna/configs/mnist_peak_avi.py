"""peak-extraction campaign for mnist on aviamala.

full sweep: all 18 peak methods x 4 alpha slices = 72 studies total.
per-method target_trials follows linear 32*d_eff rule (d_eff = active
suggest_* calls + 0.5 per conditional). hyperband min=900, max=24300, rf=3
unchanged from the per-method-tier scaffold; only the dispatch budget moves.

dormant until `bash submit_keeper.sh --config ex.utils.hpo.optuna.configs.mnist_peak_avi ...`
"""
from ex.utils.hpo.optuna.study_config import StudyConfig

CONFIG = StudyConfig(
    study_seed=1729,
    experiment="mnist",
    methods=[
        # cls family
        "BDRE_peak",
        "MDRE_peak",
        "MultiHeadTDRE_peak",
        "TriangularMDRE_peak",
        "MultiHeadTriangularTDRE_peak",
        # reg flat
        "TSM_peak",
        "CTSM_peak",
        "VFM_peak",
        "FMDRE_peak",
        "FMDRE_S2_peak",
        # triangular reg
        "TriangularTSM_peak",
        "TriangularCTSM_V1_peak",
        "TriangularCTSM_V2_peak",
        "TriangularCTSM_V3_peak",
        "TriangularVFM_V1_peak",
        "TriangularVFM_V2_peak",
        "TriangularVFM_V3_peak",
        "TriangularFMDRE_peak",
    ],
    target_trials={
        # linear 32 * d_eff (d_eff counts active suggest_* calls + 0.5/conditional)
        "BDRE_peak": 128,                       # d=4
        "MDRE_peak": 160,                       # d=5
        "MultiHeadTDRE_peak": 224,              # d=7
        "TriangularMDRE_peak": 256,             # d=8
        "MultiHeadTriangularTDRE_peak": 320,    # d=10
        "TriangularVFM_V3_peak": 352,           # d=11
        "TSM_peak": 368,                        # d=11.5
        "TriangularCTSM_V3_peak": 384,          # d=12
        "TriangularTSM_peak": 416,              # d=13
        "TriangularVFM_V2_peak": 432,           # d=13.5
        "FMDRE_peak": 448,                      # d=14
        "FMDRE_S2_peak": 480,                   # d=15
        "TriangularFMDRE_peak": 480,            # d=15
        "TriangularCTSM_V2_peak": 480,          # d=15
        "VFM_peak": 496,                        # d=15.5
        "TriangularVFM_V1_peak": 496,           # d=15.5
        "CTSM_peak": 512,                       # d=16
        "TriangularCTSM_V1_peak": 544,          # d=17
    },
    slices=[0, 1, 2, 3],
    walltime_minutes=480,
    min_resource=900,
    max_resource=24300,
    reduction_factor=3,
    holdout_top_k=5,
    walltime_margin_minutes=10,
    fixed_hp={"n_hidden_layers": 5},
    lanes=["general", "preempt", "cpu", "array"],
    resume_existing=True,
    include_tabular=False,
)
