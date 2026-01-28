"""
Step 4: Plot Results for Plugin DRE Experiment

Creates RGB colormap visualization with:
- Single figure with N subfigures (one per algorithm)
- Each subfigure has 2x2 quadrants (one per KL distance)
- All grid points shown for each algorithm
- Global error scaling across all algorithms and KL distances
"""
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch
from torch.distributions import MultivariateNormal
import yaml
import seaborn as sns


config = yaml.load(open('experiments/plugin_dre/config.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
RESULTS_DIR = config['results_dir']
FIGURES_DIR = config['figures_dir']
# grid parameters
GRID_SIZE = config['grid_size']
RGB_RESOLUTION = config['rgb_resolution']
MARKER_SIZE_MIN = config['marker_size_min']
MARKER_SIZE_MAX = config['marker_size_max']
MARKER_ALPHA_MIN = config['marker_alpha_min']
MARKER_ALPHA_MAX = config['marker_alpha_max']

dataset_filename = f'{DATA_DIR}/dataset.h5'
metrics_filename = f'{RESULTS_DIR}/metrics.h5'


def create_rgb_background(mu0, Sigma0, mu1, Sigma1, bounds, resolution=300):
    """Create RGB background image showing p0 (red) and p1 (blue) densities.

    Colors based on relative densities:
    - Red channel proportional to p0 density
    - Blue channel proportional to p1 density
    - Overlapping regions appear purple/magenta
    """
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    x_min, x_max, y_min, y_max = bounds
    xs = torch.linspace(x_min, x_max, resolution)
    ys = torch.linspace(y_min, y_max, resolution)
    # Create meshgrid with 'ij' indexing for proper row/col correspondence
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Compute densities
    density_p0 = torch.exp(p0.log_prob(points)).reshape(resolution, resolution)
    density_p1 = torch.exp(p1.log_prob(points)).reshape(resolution, resolution)

    # Normalize each density independently to [0, 1] using its own max
    red = density_p0 / (density_p0.max() + 1e-10)
    blue = density_p1 / (density_p1.max() + 1e-10)

    # Apply power transform to enhance visibility in tails
    red = torch.pow(red, 0.3)
    blue = torch.pow(blue, 0.3)

    rgb = torch.stack([red, torch.zeros_like(red), blue], dim=-1)
    return rgb.numpy()


def create_ldr_heatmap(mu0, Sigma0, mu1, Sigma1, bounds, resolution=300):
    """Create heatmap of true log density ratio log(p0/p1).

    Returns the LDR values and the symmetric vmax for colormap scaling.
    """
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    x_min, x_max, y_min, y_max = bounds
    xs = torch.linspace(x_min, x_max, resolution)
    ys = torch.linspace(y_min, y_max, resolution)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Compute log density ratio
    ldr = (p0.log_prob(points) - p1.log_prob(points)).reshape(resolution, resolution)
    return ldr.numpy()


def compute_global_error_bounds(abs_errors_dict):
    """Compute global min/max log-errors for consistent scaling across all algorithms and KL distances."""
    all_errors = np.concatenate([e.flatten() for e in abs_errors_dict.values()])
    log_errors = np.log(all_errors + 1e-8)
    return log_errors.min(), log_errors.max()


def scale_marker_sizes_global(errors, global_log_min, global_log_max, min_size=20, max_size=300):
    """Scale errors to marker sizes using global bounds for consistent scaling."""
    log_errors = np.log(errors + 1e-8)
    # Handle edge case where all errors are the same
    if global_log_max - global_log_min < 1e-8:
        return np.full_like(errors, (min_size + max_size) / 2)
    normalized = (log_errors - global_log_min) / (global_log_max - global_log_min)
    # Clip to [0, 1] to handle any numerical issues
    normalized = np.clip(normalized, 0, 1)
    return min_size + normalized * (max_size - min_size)


def scale_alpha_global(errors, global_min, global_max, min_alpha=0.1, max_alpha=0.9):
    """Scale errors to alpha values using raw error values (not log)."""
    if global_max - global_min < 1e-8:
        return np.full_like(errors, (min_alpha + max_alpha) / 2)
    normalized = (errors - global_min) / (global_max - global_min)
    normalized = np.clip(normalized, 0, 1)
    return min_alpha + normalized * (max_alpha - min_alpha)


# Load data
with h5py.File(dataset_filename, 'r') as f:
    kl_distances = f['kl_distance_arr'][:]
    mu0_arr = f['mu0_arr'][:]
    mu1_arr = f['mu1_arr'][:]
    Sigma0_arr = f['Sigma0_arr'][:]
    Sigma1_arr = f['Sigma1_arr'][:]
    grid_points_arr = f['grid_points_arr'][:]
    grid_bounds_arr = f['grid_bounds_arr'][:]

# Discover algorithms from metrics file
with h5py.File(metrics_filename, 'r') as f:
    alg_names = sorted([key.replace('abs_errors_', '') for key in f.keys() if key.startswith('abs_errors_')])

print(f"Found algorithms: {alg_names}")
num_algorithms = len(alg_names)
num_kls = len(kl_distances)

# Load all absolute errors
with h5py.File(metrics_filename, 'r') as f:
    abs_errors_dict = {alg: f[f'abs_errors_{alg}'][:] for alg in alg_names}

# Compute global error bounds for consistent scaling
global_log_min, global_log_max = compute_global_error_bounds(abs_errors_dict)
print(f"Global log-error range: [{global_log_min:.3f}, {global_log_max:.3f}]")

# Compute global raw error bounds for alpha scaling
all_errors = np.concatenate([e.flatten() for e in abs_errors_dict.values()])
global_raw_min, global_raw_max = all_errors.min(), all_errors.max()
print(f"Global raw error range: [{global_raw_min:.3f}, {global_raw_max:.3f}]")

# Create figure: grid of subfigures (one per algorithm), each with 2x2 KL quadrants
os.makedirs(FIGURES_DIR, exist_ok=True)

# Determine grid layout for algorithms
nrows_alg = int(np.ceil(np.sqrt(num_algorithms)))
ncols_alg = int(np.ceil(num_algorithms / nrows_alg))

# Create figure with subfigures
fig = plt.figure(figsize=(5 * ncols_alg, 5 * nrows_alg))
subfigs = fig.subfigures(nrows_alg, ncols_alg, wspace=0.05, hspace=0.1)

# Handle case where subfigs might not be 2D array (single row/col)
if num_algorithms == 1:
    subfigs = np.array([[subfigs]])
elif nrows_alg == 1:
    subfigs = subfigs.reshape(1, -1)
elif ncols_alg == 1:
    subfigs = subfigs.reshape(-1, 1)

# Precompute RGB backgrounds for each KL distance
rgb_backgrounds = {}
for kl_idx in range(num_kls):
    mu0 = torch.from_numpy(mu0_arr[kl_idx])
    Sigma0 = torch.from_numpy(Sigma0_arr[kl_idx])
    mu1 = torch.from_numpy(mu1_arr[kl_idx])
    Sigma1 = torch.from_numpy(Sigma1_arr[kl_idx])
    bounds = grid_bounds_arr[kl_idx]
    rgb_backgrounds[kl_idx] = create_rgb_background(mu0, Sigma0, mu1, Sigma1, bounds, resolution=RGB_RESOLUTION)

# KL layout in 2x2 grid: [[0.5, 2.0], [4.0, 8.0]]
kl_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # positions for KL indices 0,1,2,3

# Plot each algorithm
for alg_idx, alg_name in enumerate(alg_names):
    row_alg = alg_idx // ncols_alg
    col_alg = alg_idx % ncols_alg

    subfig = subfigs[row_alg, col_alg]
    subfig.suptitle(alg_name, fontsize=14, fontweight='bold', y=1.02)

    # Create 2x2 subplots within this subfigure (one per KL distance)
    # Use gridspec_kw to remove spacing between quadrants
    axes = subfig.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0})

    for kl_idx in range(num_kls):
        kl = kl_distances[kl_idx]
        row_kl, col_kl = kl_positions[kl_idx]
        ax = axes[row_kl, col_kl]

        # Get data for this KL
        bounds = grid_bounds_arr[kl_idx]
        grid_points = grid_points_arr[kl_idx]  # (num_grid_points, 2)

        # Display RGB background
        rgb_img = rgb_backgrounds[kl_idx]
        ax.imshow(rgb_img, origin='lower', extent=bounds, aspect='equal')

        # Get errors for this algorithm and KL
        errors = abs_errors_dict[alg_name][kl_idx]  # all grid points
        xs = grid_points[:, 0]
        ys = grid_points[:, 1]

        # Scale marker sizes using global log bounds
        sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                          MARKER_SIZE_MIN, MARKER_SIZE_MAX)

        # Scale alpha using global raw error bounds
        alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                    min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)

        # Create RGBA colors with per-point alpha (green with varying transparency)
        colors = np.zeros((len(xs), 4))
        colors[:, 1] = 1.0  # Green channel
        colors[:, 3] = alphas  # Alpha channel

        # Plot all grid points as green squares with no borders
        ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

        # Top row: KL label above, bottom row: KL label below
        if row_kl == 0:
            ax.set_title(f'KL = {kl:.1f}', fontsize=10)
        else:
            ax.set_xlabel(f'KL = {kl:.1f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

# Hide unused subfigures if num_algorithms doesn't fill grid
for idx in range(num_algorithms, nrows_alg * ncols_alg):
    row_alg = idx // ncols_alg
    col_alg = idx % ncols_alg
    subfigs[row_alg, col_alg].set_visible(False)

plt.tight_layout()
output_path = f'{FIGURES_DIR}/plugin_dre.pdf'
plt.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"Figure saved to: {output_path}")

# Also save as PNG for quick viewing
png_path = f'{FIGURES_DIR}/plugin_dre.png'
plt.savefig(png_path, bbox_inches='tight', dpi=150)
print(f"PNG saved to: {png_path}")

plt.close()

# ============================================================================
# Second figure: LDR heatmap background
# ============================================================================
print("\nCreating LDR heatmap figure...")

# Precompute LDR heatmaps for each KL distance
ldr_heatmaps = {}
for kl_idx in range(num_kls):
    mu0 = torch.from_numpy(mu0_arr[kl_idx])
    Sigma0 = torch.from_numpy(Sigma0_arr[kl_idx])
    mu1 = torch.from_numpy(mu1_arr[kl_idx])
    Sigma1 = torch.from_numpy(Sigma1_arr[kl_idx])
    bounds = grid_bounds_arr[kl_idx]
    ldr_heatmaps[kl_idx] = create_ldr_heatmap(mu0, Sigma0, mu1, Sigma1, bounds, resolution=RGB_RESOLUTION)

# Compute global LDR range for consistent colormap scaling
all_ldrs = np.concatenate([ldr.flatten() for ldr in ldr_heatmaps.values()])
ldr_min, ldr_max = all_ldrs.min(), all_ldrs.max()
print(f"Global LDR range: [{ldr_min:.3f}, {ldr_max:.3f}]")

# Use seaborn's icefire colormap (diverging, handles extreme values well)
ldr_cmap = sns.color_palette("icefire", as_cmap=True)

# Create second figure with subfigures
fig2 = plt.figure(figsize=(5 * ncols_alg, 5 * nrows_alg))
subfigs2 = fig2.subfigures(nrows_alg, ncols_alg, wspace=0.05, hspace=0.1)

# Handle case where subfigs might not be 2D array
if num_algorithms == 1:
    subfigs2 = np.array([[subfigs2]])
elif nrows_alg == 1:
    subfigs2 = subfigs2.reshape(1, -1)
elif ncols_alg == 1:
    subfigs2 = subfigs2.reshape(-1, 1)

# Plot each algorithm
for alg_idx, alg_name in enumerate(alg_names):
    row_alg = alg_idx // ncols_alg
    col_alg = alg_idx % ncols_alg

    subfig = subfigs2[row_alg, col_alg]
    subfig.suptitle(alg_name, fontsize=14, fontweight='bold', y=1.02)

    axes = subfig.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0})

    for kl_idx in range(num_kls):
        kl = kl_distances[kl_idx]
        row_kl, col_kl = kl_positions[kl_idx]
        ax = axes[row_kl, col_kl]

        bounds = grid_bounds_arr[kl_idx]
        grid_points = grid_points_arr[kl_idx]

        # Display LDR heatmap with asymmetric scaling around zero
        ldr_img = ldr_heatmaps[kl_idx]
        im = ax.imshow(ldr_img, origin='lower', extent=bounds, aspect='equal',
                       cmap=ldr_cmap, norm=TwoSlopeNorm(vmin=ldr_min, vcenter=0, vmax=ldr_max))

        # Get errors for this algorithm and KL
        errors = abs_errors_dict[alg_name][kl_idx]
        xs = grid_points[:, 0]
        ys = grid_points[:, 1]

        # Scale marker sizes using global log bounds
        sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                          MARKER_SIZE_MIN, MARKER_SIZE_MAX)

        # Scale alpha using global raw error bounds
        alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                    min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)

        # Create RGBA colors (green squares)
        colors = np.zeros((len(xs), 4))
        colors[:, 1] = 1.0
        colors[:, 3] = alphas

        ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

        if row_kl == 0:
            ax.set_title(f'KL = {kl:.1f}', fontsize=10)
        else:
            ax.set_xlabel(f'KL = {kl:.1f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

# Hide unused subfigures
for idx in range(num_algorithms, nrows_alg * ncols_alg):
    row_alg = idx // ncols_alg
    col_alg = idx % ncols_alg
    subfigs2[row_alg, col_alg].set_visible(False)

# Leave space on right for colorbar, then add it
fig2.subplots_adjust(right=0.88)
cbar_ax = fig2.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig2.colorbar(im, cax=cbar_ax)
cbar.set_label('Log Density Ratio (log p₀/p₁)', fontsize=12)
output_path2 = f'{FIGURES_DIR}/plugin_dre_ldr.pdf'
plt.savefig(output_path2, bbox_inches='tight', dpi=150)
print(f"LDR figure saved to: {output_path2}")

png_path2 = f'{FIGURES_DIR}/plugin_dre_ldr.png'
plt.savefig(png_path2, bbox_inches='tight', dpi=150)
print(f"LDR PNG saved to: {png_path2}")

plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("-" * 60)
with h5py.File(metrics_filename, 'r') as f:
    for alg_name in alg_names:
        mae = f[f'mae_{alg_name}'][:]
        print(f"\n{alg_name}:")
        for i, kl in enumerate(kl_distances):
            print(f"  KL={kl:.1f}: MAE={mae[i]:.4f}")
