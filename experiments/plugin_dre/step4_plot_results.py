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


def get_2d_subsample_indices(grid_size, step=2):
    """Get indices for 2D subsampling of a flattened grid array."""
    indices = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)[::step, ::step].flatten()
    return indices


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

# Standard order: BDRE -> TDRE -> MDRE -> TSM -> TriangularMDRE -> VFM
algorithm_order = ["BDRE", "TDRE_5", "MDRE_15", "TSM", "TriangularMDRE", "VFM"]

# Discover algorithms from metrics file and order them
with h5py.File(metrics_filename, 'r') as f:
    available_algs = set([key.replace('abs_errors_', '') for key in f.keys() if key.startswith('abs_errors_') and 'TriangularTDRE' not in key])
    alg_names = [alg for alg in algorithm_order if alg in available_algs]

print(f"Found algorithms: {alg_names}")
num_algorithms = len(alg_names)
num_kls = len(kl_distances)

# Load all absolute errors and MAE values
with h5py.File(metrics_filename, 'r') as f:
    abs_errors_dict = {alg: f[f'abs_errors_{alg}'][:] for alg in alg_names}
    mae_dict = {alg: f[f'mae_{alg}'][:] for alg in alg_names}

# Compute global error bounds for consistent scaling
global_log_min, global_log_max = compute_global_error_bounds(abs_errors_dict)
print(f"Global log-error range: [{global_log_min:.3f}, {global_log_max:.3f}]")

# Compute global raw error bounds for alpha scaling
all_errors = np.concatenate([e.flatten() for e in abs_errors_dict.values()])
global_raw_min, global_raw_max = all_errors.min(), all_errors.max()
print(f"Global raw error range: [{global_raw_min:.3f}, {global_raw_max:.3f}]")

# Create figure: grid of subfigures (one per algorithm), each with 2x2 KL quadrants
os.makedirs(FIGURES_DIR, exist_ok=True)

# Determine grid layout for algorithms (prefer wider layouts)
ncols_alg = int(np.ceil(np.sqrt(num_algorithms)))
nrows_alg = int(np.ceil(num_algorithms / ncols_alg))

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
    subfig.suptitle(alg_name, fontsize=14, fontweight='bold', y=0.98)

    # Create 2x2 subplots within this subfigure (one per KL distance)
    # Use gridspec_kw to remove spacing between quadrants
    axes = subfig.subplots(2, 2)
    subfig.subplots_adjust(wspace=0, hspace=0)

    for kl_idx in range(num_kls):
        kl = kl_distances[kl_idx]
        row_kl, col_kl = kl_positions[kl_idx]
        ax = axes[row_kl, col_kl]

        # Get data for this KL
        bounds = grid_bounds_arr[kl_idx]
        grid_points = grid_points_arr[kl_idx]  # (num_grid_points, 2)

        # Display RGB background
        rgb_img = rgb_backgrounds[kl_idx]
        ax.imshow(rgb_img, origin='lower', extent=bounds, aspect='auto')

        # Get errors for this algorithm and KL (2D subsample for multi-KL plots)
        subsample_idx = get_2d_subsample_indices(GRID_SIZE, step=2)
        errors = abs_errors_dict[alg_name][kl_idx][subsample_idx]
        xs = grid_points[subsample_idx, 0]
        ys = grid_points[subsample_idx, 1]

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

        # Plot subsampled grid points as green circles with no borders
        ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

        # Top row: KL label above, bottom row: KL label below (with MAE)
        mae_this_kl = mae_dict[alg_name][kl_idx]
        label = f'KL = {kl:.1f} (MAE: {mae_this_kl:.3f})'
        if row_kl == 0:
            ax.set_title(label, fontsize=10)
        else:
            ax.set_xlabel(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

# Hide unused subfigures if num_algorithms doesn't fill grid
for idx in range(num_algorithms, nrows_alg * ncols_alg):
    row_alg = idx // ncols_alg
    col_alg = idx % ncols_alg
    subfigs[row_alg, col_alg].set_visible(False)

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
    subfig.suptitle(alg_name, fontsize=14, fontweight='bold', y=0.98)

    axes = subfig.subplots(2, 2)
    subfig.subplots_adjust(wspace=0, hspace=0)

    for kl_idx in range(num_kls):
        kl = kl_distances[kl_idx]
        row_kl, col_kl = kl_positions[kl_idx]
        ax = axes[row_kl, col_kl]

        bounds = grid_bounds_arr[kl_idx]
        grid_points = grid_points_arr[kl_idx]

        # Display LDR heatmap with asymmetric scaling around zero
        ldr_img = ldr_heatmaps[kl_idx]
        im = ax.imshow(ldr_img, origin='lower', extent=bounds, aspect='auto',
                       cmap=ldr_cmap, norm=TwoSlopeNorm(vmin=ldr_min, vcenter=0, vmax=ldr_max))

        # Get errors for this algorithm and KL (2D subsample for multi-KL plots)
        subsample_idx = get_2d_subsample_indices(GRID_SIZE, step=2)
        errors = abs_errors_dict[alg_name][kl_idx][subsample_idx]
        xs = grid_points[subsample_idx, 0]
        ys = grid_points[subsample_idx, 1]

        # Scale marker sizes using global log bounds
        sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                          MARKER_SIZE_MIN, MARKER_SIZE_MAX)

        # Scale alpha using global raw error bounds
        alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                    min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)

        # Create RGBA colors (green circles)
        colors = np.zeros((len(xs), 4))
        colors[:, 1] = 1.0
        colors[:, 3] = alphas

        ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

        # Top row: KL label above, bottom row: KL label below (with MAE)
        mae_this_kl = mae_dict[alg_name][kl_idx]
        label = f'KL = {kl:.1f} (MAE: {mae_this_kl:.3f})'
        if row_kl == 0:
            ax.set_title(label, fontsize=10)
        else:
            ax.set_xlabel(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

# Hide unused subfigures
for idx in range(num_algorithms, nrows_alg * ncols_alg):
    row_alg = idx // ncols_alg
    col_alg = idx % ncols_alg
    subfigs2[row_alg, col_alg].set_visible(False)

# Leave space on right for colorbar, then add it
fig2.subplots_adjust(right=0.85)
cbar_ax = fig2.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig2.colorbar(im, cax=cbar_ax)
cbar.set_label('Log Density Ratio (log p₀/p₁)', fontsize=12)
output_path2 = f'{FIGURES_DIR}/plugin_dre_ldr.pdf'
plt.savefig(output_path2, bbox_inches='tight', dpi=150)
print(f"LDR figure saved to: {output_path2}")

png_path2 = f'{FIGURES_DIR}/plugin_dre_ldr.png'
plt.savefig(png_path2, bbox_inches='tight', dpi=150)
print(f"LDR PNG saved to: {png_path2}")

plt.close()

# ============================================================================
# Third figure: RGB background, largest KL only (single panel per algorithm)
# ============================================================================
print("\nCreating single-panel RGB figure (largest KL only)...")

largest_kl_idx = num_kls - 1
largest_kl = kl_distances[largest_kl_idx]

fig3 = plt.figure(figsize=(4 * ncols_alg, 4 * nrows_alg))
axes3 = fig3.subplots(nrows_alg, ncols_alg)

# Handle 1D/2D array cases for axes3
if num_algorithms == 1:
    axes3 = np.array([[axes3]])
elif nrows_alg == 1:
    axes3 = axes3.reshape(1, -1)
elif ncols_alg == 1:
    axes3 = axes3.reshape(-1, 1)

for alg_idx, alg_name in enumerate(alg_names):
    row_alg = alg_idx // ncols_alg
    col_alg = alg_idx % ncols_alg
    ax = axes3[row_alg, col_alg]

    bounds = grid_bounds_arr[largest_kl_idx]
    grid_points = grid_points_arr[largest_kl_idx]

    # RGB background
    ax.imshow(rgb_backgrounds[largest_kl_idx], origin='lower', extent=bounds, aspect='auto')

    # Errors and scatter
    errors = abs_errors_dict[alg_name][largest_kl_idx]
    xs, ys = grid_points[:, 0], grid_points[:, 1]
    sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                      MARKER_SIZE_MIN, MARKER_SIZE_MAX)
    alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)
    colors = np.zeros((len(xs), 4))
    colors[:, 1] = 1.0
    colors[:, 3] = alphas
    ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

    # Title with MAE
    mae_this_kl = mae_dict[alg_name][largest_kl_idx]
    ax.set_title(f'{alg_name} (MAE: {mae_this_kl:.3f})', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# Hide unused subplots
for idx in range(num_algorithms, nrows_alg * ncols_alg):
    row_alg = idx // ncols_alg
    col_alg = idx % ncols_alg
    axes3[row_alg, col_alg].set_visible(False)

plt.tight_layout()
output_path3 = f'{FIGURES_DIR}/plugin_dre_largest_kl.pdf'
plt.savefig(output_path3, bbox_inches='tight', dpi=150)
print(f"Largest KL RGB figure saved to: {output_path3}")

png_path3 = f'{FIGURES_DIR}/plugin_dre_largest_kl.png'
plt.savefig(png_path3, bbox_inches='tight', dpi=150)
print(f"Largest KL RGB PNG saved to: {png_path3}")

plt.close()

# ============================================================================
# Fourth figure: LDR heatmap, largest KL only (single panel per algorithm)
# ============================================================================
print("\nCreating single-panel LDR figure (largest KL only)...")

fig4 = plt.figure(figsize=(4 * ncols_alg, 4 * nrows_alg))
axes4 = fig4.subplots(nrows_alg, ncols_alg)

# Handle 1D/2D array cases for axes4
if num_algorithms == 1:
    axes4 = np.array([[axes4]])
elif nrows_alg == 1:
    axes4 = axes4.reshape(1, -1)
elif ncols_alg == 1:
    axes4 = axes4.reshape(-1, 1)

for alg_idx, alg_name in enumerate(alg_names):
    row_alg = alg_idx // ncols_alg
    col_alg = alg_idx % ncols_alg
    ax = axes4[row_alg, col_alg]

    bounds = grid_bounds_arr[largest_kl_idx]
    grid_points = grid_points_arr[largest_kl_idx]

    # LDR heatmap background
    ldr_img = ldr_heatmaps[largest_kl_idx]
    im = ax.imshow(ldr_img, origin='lower', extent=bounds, aspect='auto',
                   cmap=ldr_cmap, norm=TwoSlopeNorm(vmin=ldr_min, vcenter=0, vmax=ldr_max))

    # Errors and scatter
    errors = abs_errors_dict[alg_name][largest_kl_idx]
    xs, ys = grid_points[:, 0], grid_points[:, 1]
    sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                      MARKER_SIZE_MIN, MARKER_SIZE_MAX)
    alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)
    colors = np.zeros((len(xs), 4))
    colors[:, 1] = 1.0
    colors[:, 3] = alphas
    ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

    # Title with MAE
    mae_this_kl = mae_dict[alg_name][largest_kl_idx]
    ax.set_title(f'{alg_name} (MAE: {mae_this_kl:.3f})', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# Hide unused subplots
for idx in range(num_algorithms, nrows_alg * ncols_alg):
    row_alg = idx // ncols_alg
    col_alg = idx % ncols_alg
    axes4[row_alg, col_alg].set_visible(False)

# Add colorbar
fig4.subplots_adjust(right=0.85)
cbar_ax4 = fig4.add_axes([0.88, 0.15, 0.02, 0.7])
cbar4 = fig4.colorbar(im, cax=cbar_ax4)
cbar4.set_label('Log Density Ratio (log p₀/p₁)', fontsize=12)

output_path4 = f'{FIGURES_DIR}/plugin_dre_ldr_largest_kl.pdf'
plt.savefig(output_path4, bbox_inches='tight', dpi=150)
print(f"Largest KL LDR figure saved to: {output_path4}")

png_path4 = f'{FIGURES_DIR}/plugin_dre_ldr_largest_kl.png'
plt.savefig(png_path4, bbox_inches='tight', dpi=150)
print(f"Largest KL LDR PNG saved to: {png_path4}")

plt.close()

# ============================================================================
# Fifth figure: RGB background, 2x2 per algorithm, 1xN layout (single row)
# ============================================================================
print("\nCreating 1xN RGB figure (all algorithms in single row)...")

fig5 = plt.figure(figsize=(4.5 * num_algorithms, 5))
subfigs5 = fig5.subfigures(1, num_algorithms, wspace=0)

# Handle single algorithm case
if num_algorithms == 1:
    subfigs5 = [subfigs5]

for alg_idx, alg_name in enumerate(alg_names):
    subfig = subfigs5[alg_idx]
    subfig.suptitle(alg_name, fontsize=14, fontweight='bold', y=0.98)

    axes = subfig.subplots(2, 2)
    subfig.subplots_adjust(wspace=0, hspace=0)

    for kl_idx in range(num_kls):
        kl = kl_distances[kl_idx]
        row_kl, col_kl = kl_positions[kl_idx]
        ax = axes[row_kl, col_kl]

        bounds = grid_bounds_arr[kl_idx]
        grid_points = grid_points_arr[kl_idx]

        rgb_img = rgb_backgrounds[kl_idx]
        ax.imshow(rgb_img, origin='lower', extent=bounds, aspect='auto')

        # 2D subsample for multi-KL plots
        subsample_idx = get_2d_subsample_indices(GRID_SIZE, step=2)
        errors = abs_errors_dict[alg_name][kl_idx][subsample_idx]
        xs = grid_points[subsample_idx, 0]
        ys = grid_points[subsample_idx, 1]

        sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                          MARKER_SIZE_MIN, MARKER_SIZE_MAX)
        alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                    min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)

        colors = np.zeros((len(xs), 4))
        colors[:, 1] = 1.0
        colors[:, 3] = alphas

        ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

        mae_this_kl = mae_dict[alg_name][kl_idx]
        label = f'KL = {kl:.1f} (MAE: {mae_this_kl:.3f})'
        if row_kl == 0:
            ax.set_title(label, fontsize=10)
        else:
            ax.set_xlabel(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

output_path5 = f'{FIGURES_DIR}/plugin_dre_wide.pdf'
plt.savefig(output_path5, bbox_inches='tight', dpi=150)
print(f"Wide RGB figure saved to: {output_path5}")

png_path5 = f'{FIGURES_DIR}/plugin_dre_wide.png'
plt.savefig(png_path5, bbox_inches='tight', dpi=150)
print(f"Wide RGB PNG saved to: {png_path5}")

plt.close()

# ============================================================================
# Sixth figure: LDR heatmap, 2x2 per algorithm, 1xN layout (single row)
# ============================================================================
print("\nCreating 1xN LDR figure (all algorithms in single row)...")

fig6 = plt.figure(figsize=(4.5 * num_algorithms, 5))
subfigs6 = fig6.subfigures(1, num_algorithms, wspace=0)

if num_algorithms == 1:
    subfigs6 = [subfigs6]

for alg_idx, alg_name in enumerate(alg_names):
    subfig = subfigs6[alg_idx]
    subfig.suptitle(alg_name, fontsize=14, fontweight='bold', y=0.98)

    axes = subfig.subplots(2, 2)
    subfig.subplots_adjust(wspace=0, hspace=0)

    for kl_idx in range(num_kls):
        kl = kl_distances[kl_idx]
        row_kl, col_kl = kl_positions[kl_idx]
        ax = axes[row_kl, col_kl]

        bounds = grid_bounds_arr[kl_idx]
        grid_points = grid_points_arr[kl_idx]

        ldr_img = ldr_heatmaps[kl_idx]
        im = ax.imshow(ldr_img, origin='lower', extent=bounds, aspect='auto',
                       cmap=ldr_cmap, norm=TwoSlopeNorm(vmin=ldr_min, vcenter=0, vmax=ldr_max))

        # 2D subsample for multi-KL plots
        subsample_idx = get_2d_subsample_indices(GRID_SIZE, step=2)
        errors = abs_errors_dict[alg_name][kl_idx][subsample_idx]
        xs = grid_points[subsample_idx, 0]
        ys = grid_points[subsample_idx, 1]

        sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                          MARKER_SIZE_MIN, MARKER_SIZE_MAX)
        alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                    min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)

        colors = np.zeros((len(xs), 4))
        colors[:, 1] = 1.0
        colors[:, 3] = alphas

        ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

        mae_this_kl = mae_dict[alg_name][kl_idx]
        label = f'KL = {kl:.1f} (MAE: {mae_this_kl:.3f})'
        if row_kl == 0:
            ax.set_title(label, fontsize=10)
        else:
            ax.set_xlabel(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

fig6.subplots_adjust(right=0.92)
cbar_ax6 = fig6.add_axes([0.94, 0.15, 0.01, 0.7])
cbar6 = fig6.colorbar(im, cax=cbar_ax6)
cbar6.set_label('Log Density Ratio (log p₀/p₁)', fontsize=12)

output_path6 = f'{FIGURES_DIR}/plugin_dre_ldr_wide.pdf'
plt.savefig(output_path6, bbox_inches='tight', dpi=150)
print(f"Wide LDR figure saved to: {output_path6}")

png_path6 = f'{FIGURES_DIR}/plugin_dre_ldr_wide.png'
plt.savefig(png_path6, bbox_inches='tight', dpi=150)
print(f"Wide LDR PNG saved to: {png_path6}")

plt.close()

# ============================================================================
# Seventh figure: RGB background, largest KL only, 1xN layout (single row)
# ============================================================================
print("\nCreating 1xN RGB figure (largest KL only, single row)...")

fig7 = plt.figure(figsize=(4 * num_algorithms, 4))
axes7 = fig7.subplots(1, num_algorithms)

if num_algorithms == 1:
    axes7 = [axes7]

for alg_idx, alg_name in enumerate(alg_names):
    ax = axes7[alg_idx]

    bounds = grid_bounds_arr[largest_kl_idx]
    grid_points = grid_points_arr[largest_kl_idx]

    ax.imshow(rgb_backgrounds[largest_kl_idx], origin='lower', extent=bounds, aspect='auto')

    errors = abs_errors_dict[alg_name][largest_kl_idx]
    xs, ys = grid_points[:, 0], grid_points[:, 1]
    sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                      MARKER_SIZE_MIN, MARKER_SIZE_MAX)
    alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)
    colors = np.zeros((len(xs), 4))
    colors[:, 1] = 1.0
    colors[:, 3] = alphas
    ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

    mae_this_kl = mae_dict[alg_name][largest_kl_idx]
    ax.set_title(f'{alg_name} (MAE: {mae_this_kl:.3f})', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
output_path7 = f'{FIGURES_DIR}/plugin_dre_largest_kl_wide.pdf'
plt.savefig(output_path7, bbox_inches='tight', dpi=150)
print(f"Wide largest KL RGB figure saved to: {output_path7}")

png_path7 = f'{FIGURES_DIR}/plugin_dre_largest_kl_wide.png'
plt.savefig(png_path7, bbox_inches='tight', dpi=150)
print(f"Wide largest KL RGB PNG saved to: {png_path7}")

plt.close()

# ============================================================================
# Eighth figure: LDR heatmap, largest KL only, 1xN layout (single row)
# ============================================================================
print("\nCreating 1xN LDR figure (largest KL only, single row)...")

fig8 = plt.figure(figsize=(4 * num_algorithms, 4))
axes8 = fig8.subplots(1, num_algorithms)

if num_algorithms == 1:
    axes8 = [axes8]

for alg_idx, alg_name in enumerate(alg_names):
    ax = axes8[alg_idx]

    bounds = grid_bounds_arr[largest_kl_idx]
    grid_points = grid_points_arr[largest_kl_idx]

    ldr_img = ldr_heatmaps[largest_kl_idx]
    im = ax.imshow(ldr_img, origin='lower', extent=bounds, aspect='auto',
                   cmap=ldr_cmap, norm=TwoSlopeNorm(vmin=ldr_min, vcenter=0, vmax=ldr_max))

    errors = abs_errors_dict[alg_name][largest_kl_idx]
    xs, ys = grid_points[:, 0], grid_points[:, 1]
    sizes = scale_marker_sizes_global(errors, global_log_min, global_log_max,
                                      MARKER_SIZE_MIN, MARKER_SIZE_MAX)
    alphas = scale_alpha_global(errors, global_raw_min, global_raw_max,
                                min_alpha=MARKER_ALPHA_MIN, max_alpha=MARKER_ALPHA_MAX)
    colors = np.zeros((len(xs), 4))
    colors[:, 1] = 1.0
    colors[:, 3] = alphas
    ax.scatter(xs, ys, s=sizes, c=colors, marker='s', edgecolors='none')

    mae_this_kl = mae_dict[alg_name][largest_kl_idx]
    ax.set_title(f'{alg_name} (MAE: {mae_this_kl:.3f})', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

fig8.subplots_adjust(right=0.92)
cbar_ax8 = fig8.add_axes([0.94, 0.15, 0.01, 0.7])
cbar8 = fig8.colorbar(im, cax=cbar_ax8)
cbar8.set_label('Log Density Ratio (log p₀/p₁)', fontsize=12)

output_path8 = f'{FIGURES_DIR}/plugin_dre_ldr_largest_kl_wide.pdf'
plt.savefig(output_path8, bbox_inches='tight', dpi=150)
print(f"Wide largest KL LDR figure saved to: {output_path8}")

png_path8 = f'{FIGURES_DIR}/plugin_dre_ldr_largest_kl_wide.png'
plt.savefig(png_path8, bbox_inches='tight', dpi=150)
print(f"Wide largest KL LDR PNG saved to: {png_path8}")

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
