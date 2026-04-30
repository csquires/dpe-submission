import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


sns.set_style('whitegrid')
plt.style.use('full-width.mplstyle')

config = yaml.load(open('experiments/elbo_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
ALPHAS = config['alphas']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/errors_d={DATA_DIM},nsamples={NSAMPLES}.h5'

colors = {
    # base methods
    "BDRE": "#1f77b4",
    "TDRE": "#ff7f0e",
    "MDRE": "#2ca02c",
    "TSM": "#d62728",
    "VFM": "#9467bd",
    "TriangularMDRE": "#aec7e8",
    # named variants
    "MDRE_15": "#2ca02c",
    "TDRE_5": "#ff7f0e",
    # ctsm family
    "CTSM": "#e377c2",
    "TriangularCTSM_V1": "#e377c2",
    "TriangularCTSM_V2": "#f7b6d2",
    "TriangularCTSM_V3": "#c5b0d5",
    # fmdre family
    "FMDRE": "#7f7f7f",
    "FMDRE_S2": "#c7c7c7",
    "TriangularFMDRE": "#b5a8c0",
    # triangular tsm
    "TriangularTSM": "#ff9897",
    # triangular vfm variants
    "TriangularVFM_V1": "#9467bd",
    "TriangularVFM_V2": "#c5b0d5",
    "TriangularVFM_V3": "#bcbd22",
    # multi-head variant
    "MultiHeadTriangularTDRE": "#17becf",
    # aliases (for backward compatibility)
    "MHTTDRE": "#17becf",
    "TriangularCTSM": "#e377c2",
    "TriangularVFM": "#9467bd",
}
legend_order = [
    "BDRE", "TDRE", "MDRE", "TSM", "VFM",
    "TriangularMDRE",
    "MDRE_15", "TDRE_5",
    "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
    "FMDRE", "FMDRE_S2", "TriangularFMDRE",
    "TriangularTSM",
    "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3",
    "MultiHeadTriangularTDRE",
]

# Load processed results
with h5py.File(processed_results_filename, 'r') as f:
    # Find all algorithms
    alg_names = [key.replace('mae_', '') for key in f.keys() if key.startswith('mae_')]
    mae_by_alg = {alg_name: f[f'mae_{alg_name}'][:] for alg_name in alg_names}

# Plot
plt.clf()
fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=len(ALPHAS))
all_vals = []
for mae in mae_by_alg.values():
    all_vals.extend(mae.flatten().tolist())
if all_vals:
    y_min, y_max = min(all_vals), max(all_vals)
else:
    y_min, y_max = 0.0, 1.0

for i, alpha in enumerate(ALPHAS):
    # stable legend order
    alg_names = sorted(mae_by_alg.keys(), key=lambda n: legend_order.index(n) if n in legend_order else len(legend_order))
    for alg_name in alg_names:
        mae = mae_by_alg[alg_name]
        axes[i].plot(
            DESIGN_EIG_PERCENTAGES,
            mae[:, i],
            label=alg_name,
            color=colors.get(alg_name, f"#{hash(alg_name) % 0xffffff:06x}"),
            linewidth=1.0,
        )
    axes[i].set_title(fr"$\alpha = {alpha}$")
    axes[i].set_ylim(y_min * 0.9, y_max * 1.1)
    if i != 0:
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="y", labelleft=False)

fig.supxlabel(r"$\beta$ (Design Optimality Percentage)")
fig.supylabel(r"ELBO Estimation Error")
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [lbl for lbl in legend_order if lbl in by_label]
ordered_labels += [lbl for lbl in by_label.keys() if lbl not in ordered_labels]
plt.legend([by_label[lbl] for lbl in ordered_labels], ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# saving
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(f'{FIGURES_DIR}/elbo_estimation_complete.pdf')
