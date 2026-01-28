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

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/mae_by_beta_d={DATA_DIM},nsamples={NSAMPLES}.h5'
with h5py.File(processed_results_filename, 'r') as f:
    design_eig_percentages = f['design_eig_percentages'][:]
    mae_by_beta = {key.replace('mae_by_beta_', ''): f[key][:] for key in f.keys() if key.startswith('mae_by_beta_')}

colors = {
    "BDRE": "#1f77b4",
    "TDRE_5": "#ff7f0e",
    "MDRE_15": "#2ca02c",
    "TSM": "#d62728",
    "TriangularTSM": "#17becf",
    "VFM": "#9467bd",
    "Direct3": "#8c564b",
}

plt.clf()
sns.set_style('whitegrid')
plt.style.use('half-width.mplstyle')
label_map = {
    "TDRE_5": "TDRE(5)",
    "MDRE_15": "MDRE(15)",
    "VFM": "VFM",
    "Direct3": "Direct3",
}

for alg_name, maes in mae_by_beta.items():
    if alg_name in ['Direct3', 'VFM']:
        continue
    color = colors.get(alg_name, None)
    label = label_map.get(alg_name, alg_name)
    plt.plot(design_eig_percentages, maes, label=label, color=color)
plt.xlabel(r"$\beta$ (Design Optimality Percentage)")
plt.ylabel(r"EIG Estimation Error")
plt.legend(loc="upper left", fontsize=10)
plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(f'{FIGURES_DIR}/eig_estimation.pdf')
