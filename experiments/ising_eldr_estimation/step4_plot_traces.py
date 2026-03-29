import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


config = yaml.load(open('experiments/ising_eldr_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
GRID_SIZE = config['grid_size']
NSAMPLES = config['nsamples']

# compute data_dim from grid_size (flattened ising lattice)
DATA_DIM = GRID_SIZE * GRID_SIZE

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/eldr_estimates_d={DATA_DIM},nsamples={NSAMPLES}.h5'

# load processed results
with h5py.File(processed_results_filename, 'r') as f:
    design_parameters = f['design_parameters'][:]
    true_eldrs = f['true_eldrs'][:]
    est_keys = [key for key in f.keys() if key.startswith('eldr_estimates_')]
    eldrs_by_alg = {key.replace('eldr_estimates_', ''): f[key][:] for key in est_keys}

# aggregate across priors: [num_designs, num_priors] -> [num_designs]
mean_eldrs_by_alg = {
    alg_name: estimates.mean(axis=1)
    for alg_name, estimates in eldrs_by_alg.items()
}

# color scheme and legend ordering
colors = {
    "BDRE": "#1f77b4",
    "TDRE": "#ff7f0e",
    "MDRE": "#2ca02c",
    "TSM": "#d62728",
    "TriangularMDRE": "#aec7e8",
    "VFM": "#9467bd",
}
legend_order = ["BDRE", "TDRE", "MDRE", "TSM", "TriangularMDRE", "VFM"]

# plot
plt.clf()
sns.set_style('whitegrid')
plt.style.use('half-width.mplstyle')

# plot algorithm traces
for alg_name in legend_order:
    if alg_name not in mean_eldrs_by_alg:
        continue
    plt.plot(
        design_parameters,
        mean_eldrs_by_alg[alg_name],
        label=alg_name,
        color=colors.get(alg_name, "#333333"),
        linewidth=1.0
    )

# plot ground truth last (on top)
plt.plot(
    design_parameters,
    true_eldrs,
    label="Ground Truth",
    color="#000000",
    linestyle='--',
    linewidth=2.0
)

plt.xlabel("Design Parameter")
plt.ylabel("ELDR Estimate")

# ordered legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [lbl for lbl in legend_order if lbl in by_label]
ordered_labels += [lbl for lbl in by_label.keys() if lbl not in ordered_labels]
plt.legend(
    [by_label[lbl] for lbl in ordered_labels],
    ordered_labels,
    loc="upper left",
    fontsize=16
)

plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(f'{FIGURES_DIR}/eldr_traces.pdf')
