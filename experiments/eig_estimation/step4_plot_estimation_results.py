import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import h5py


config = yaml.load(open('experiments/eig_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']

#processed_results_filename = f'{PROCESSED_RESULTS_DIR}/mae_by_beta_d={DATA_DIM},nsamples={NSAMPLES}.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/updated.h5'
with h5py.File(processed_results_filename, 'r') as f:
    design_eig_percentages = f['design_eig_percentages'][:]
    mae_by_beta = {key.replace('mae_by_beta_', ''): f[key][:] for key in f.keys() if key.startswith('mae_by_beta_')}

colors = {
    "BDRE": "#1f77b4",
    "TDRE": "#ff7f0e",
    "MDRE": "#2ca02c",
    "TSM": "#d62728",
    "TriangularMDRE": "#9467bd",
}
legend_order = ["BDRE", "TDRE", "MDRE", "TSM", "TriangularMDRE"]

plt.clf()
sns.set_style('whitegrid')
plt.style.use('half-width.mplstyle')
label_map = {
    "TDRE": "TDRE",
    "MDRE": "MDRE",
    "TriangularMDRE": "TriangularMDRE",
}

for alg_name, maes in mae_by_beta.items():
    if alg_name in ['Direct3', 'VFM', 'TriangularTSM']:
        continue
    color = colors.get(alg_name, None)
    label = label_map.get(alg_name, alg_name)
    plt.plot(design_eig_percentages, maes, label=label, color=color, linewidth=1.0)
plt.xlabel(r"$\beta$ (Design Optimality Percentage)")
plt.ylabel(r"EIG Estimation Error")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [lbl for lbl in legend_order if lbl in by_label]
ordered_labels += [lbl for lbl in by_label.keys() if lbl not in ordered_labels]
plt.legend([by_label[lbl] for lbl in ordered_labels], ordered_labels, loc="upper left", fontsize=10)
plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(f'{FIGURES_DIR}/eig_estimation_final2.pdf')
