import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import h5py


config = yaml.load(open('experiments/eig_vertex_sweep/config.yaml', 'r'), Loader=yaml.FullLoader)
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
VERTEX_WAYPOINTS = config['vertex_waypoints']

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/mae_by_beta_d={DATA_DIM},nsamples={NSAMPLES}.h5'
with h5py.File(processed_results_filename, 'r') as f:
    design_eig_percentages = f['design_eig_percentages'][:]
    mae_by_beta = {key.replace('mae_by_beta_', ''): f[key][:] for key in f.keys() if key.startswith('mae_by_beta_')}

colors = {
    "vertex_3": "#1f77b4",
    "vertex_5": "#ff7f0e",
    "vertex_7": "#2ca02c",
    "vertex_9": "#d62728",
}

label_map = {f"vertex_{v}": f"vertex = {v}" for v in VERTEX_WAYPOINTS}
legend_order = [f"vertex = {v}" for v in VERTEX_WAYPOINTS]

plt.clf()
sns.set_style('whitegrid')
plt.style.use('half-width.mplstyle')

for key, maes in mae_by_beta.items():
    color = colors.get(key, None)
    label = label_map.get(key, key)
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
plt.savefig(f'{FIGURES_DIR}/final.pdf')
