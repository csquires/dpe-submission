"""per-stratum data-card metrics + emitters for datagen diagnostics.

quantifies three data properties the standard diagnostics don't cover:
  irregularity  -- lip_q (local Lipschitz of the true LDR field), hill_tail
                   (tail index of the density ratio), hubness (embedding spaces)
  dimensionality-- twonn_id (local intrinsic dim), participation_ratio (linear
                   dim); their gap reads as manifold curvature. mean_cos for
                   embedding anisotropy.
  multimodality -- gmm_modes (BIC-selected component count), eff_modes (exact
                   exp-entropy of known mixture weights)

all metrics are scalar per cell; drivers collect them per (stratum, pair) and
hand the nested values to write_card (md + tex table, one row per stratum) and
plot_metric_boxes (one panel per metric, hardness on x, box over pairs --
same hardness-lightness language as ex/utils/family_boxplot).

    from ex.utils import data_card as dc
    vals[metric][stratum_idx] = [dc.twonn_id(X_cell) for each pair ...]
    dc.write_card(stem, strata_labels, vals)
    dc.plot_metric_boxes(vals, strata_values, sweep_name='alpha', out_dir=..., prefix=...)
"""
from __future__ import annotations

import numpy as np

from ex.utils.tables import fmt_iqr, write_tables


SUBSAMPLE = 2000
_RNG = np.random.default_rng(0)


def _sub(X, n=SUBSAMPLE):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] <= n:
        return X
    return X[_RNG.choice(X.shape[0], n, replace=False)]


# -----------------------------------------------------------------------------
# dimensionality
# -----------------------------------------------------------------------------

def twonn_id(X) -> float:
    """two-NN intrinsic dimension (Facco et al. 2017), MLE over mu = r2/r1."""
    from sklearn.neighbors import NearestNeighbors
    X = _sub(X)
    if X.shape[0] < 10:
        return np.nan
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    keep = r1 > 0
    if keep.sum() < 10:
        return np.nan
    mu = np.log(r2[keep] / r1[keep])
    mu = mu[mu > 0]
    return float(mu.size / mu.sum()) if mu.size else np.nan


def participation_ratio(X) -> float:
    """linear effective dimension (sum lam)^2 / sum lam^2 of the covariance."""
    X = _sub(X)
    lam = np.linalg.eigvalsh(np.cov(X.T))
    lam = np.clip(lam, 0, None)
    s = lam.sum()
    return float(s * s / (lam ** 2).sum()) if s > 0 else np.nan


def mean_cos(X) -> float:
    """anisotropy of an embedding space: mean pairwise cosine similarity."""
    X = _sub(X, 1000)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.clip(norm, 1e-12, None)
    G = U @ U.T
    n = G.shape[0]
    return float((G.sum() - n) / (n * (n - 1)))


# -----------------------------------------------------------------------------
# multimodality
# -----------------------------------------------------------------------------

def gmm_modes(X, kmax=8, pca_dim=20) -> float:
    """BIC-selected gaussian-mixture component count (PCA-reduced if wide)."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    X = _sub(X, 1000)
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=0).fit_transform(X)
    bics = []
    for k in range(1, kmax + 1):
        gm = GaussianMixture(k, covariance_type='diag', random_state=0,
                             n_init=1, max_iter=200).fit(X)
        bics.append(gm.bic(X))
    return float(np.argmin(bics) + 1)


def eff_modes(w) -> float:
    """exact effective mode count exp(H(w)) of known mixture weights."""
    w = np.asarray(w, dtype=np.float64)
    w = w[w > 0]
    w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


# -----------------------------------------------------------------------------
# irregularity
# -----------------------------------------------------------------------------

def lip_q(X, ldr, q=90, k=5) -> float:
    """q-th percentile of |delta ldr| / ||delta x|| over kNN pairs (LDR roughness)."""
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=np.float64)
    ldr = np.asarray(ldr, dtype=np.float64)
    keep = np.isfinite(ldr)
    X, ldr = X[keep], ldr[keep]
    if X.shape[0] > SUBSAMPLE:
        idx = _RNG.choice(X.shape[0], SUBSAMPLE, replace=False)
        X, ldr = X[idx], ldr[idx]
    if X.shape[0] < k + 1:
        return np.nan
    d, nb = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    quot = np.abs(ldr[nb[:, 1:]] - ldr[:, None]) / np.clip(d[:, 1:], 1e-12, None)
    return float(np.percentile(quot.ravel(), q))


def hill_tail(ldr, top_frac=0.05) -> float:
    """hill tail index of the density ratio r = e^ldr (smaller = heavier tail)."""
    ldr = np.asarray(ldr, dtype=np.float64)
    ldr = np.sort(ldr[np.isfinite(ldr)])[::-1]
    k = max(10, int(top_frac * ldr.size))
    if ldr.size <= k:
        return np.nan
    excess = ldr[:k] - ldr[k]
    m = excess.mean()
    return float(1.0 / m) if m > 0 else np.nan


def hubness(X, k=10) -> float:
    """skewness of the k-occurrence distribution (high-dim hubness pathology)."""
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import skew
    X = _sub(X)
    if X.shape[0] < k + 1:
        return np.nan
    _, nb = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    counts = np.bincount(nb[:, 1:].ravel(), minlength=X.shape[0])
    return float(skew(counts))


# -----------------------------------------------------------------------------
# emitters
# -----------------------------------------------------------------------------

def write_card(stem, strata_labels, values, title='Data card') -> None:
    """md + tex data card: rows = strata, cols = metrics, med [q1, q3] cells.

    values: dict metric_name -> list over strata of per-pair value lists.
    """
    metrics = list(values.keys())
    header = ['Stratum'] + metrics
    rows = []
    for si, lab in enumerate(strata_labels):
        cells = [lab]
        for m in metrics:
            v = np.asarray(values[m][si], dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                cells.append('--')
            else:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                cells.append(fmt_iqr(med, q1, q3))
        rows.append(cells)
    write_tables(stem, [(title, header, rows)])


def plot_metric_boxes(values, strata_values, *, sweep_name, out_dir, prefix,
                      panel_w=3.4, panel_h=3.4, max_cols=5) -> None:
    """one panel per metric, hardness on x, one box per stratum over pairs.

    values: dict metric_name -> list over strata of per-pair value lists.
    boxes use the family_boxplot steel-blue with the hardness lightness sweep.
    wraps to multiple rows after max_cols panels.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ex.utils.family_boxplot import COLOR_NON_TRI, _shade

    metrics = list(values.keys())
    n_sw = len(strata_values)
    fracs = np.linspace(0.45, 1.0, n_sw) if n_sw > 1 else [1.0]

    ncol = min(max_cols, len(metrics))
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                             squeeze=False)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_axis_off()
    for ax, m in zip(axes.ravel(), metrics):
        for si, sv in enumerate(strata_values):
            v = np.asarray(values[m][si], dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            bp = ax.boxplot(v, positions=[si], widths=0.55, patch_artist=True,
                            showfliers=True,
                            flierprops=dict(marker='.', markersize=3, alpha=0.4),
                            medianprops=dict(color='black', linewidth=1.2),
                            manage_ticks=False)
            bp['boxes'][0].set_facecolor(_shade(COLOR_NON_TRI, fracs[si]))
            bp['boxes'][0].set_alpha(0.7)
        ax.set_xticks(range(n_sw))
        ax.set_xticklabels([f'{v:g}' for v in strata_values], fontsize=12)
        ax.set_xlabel(sweep_name, fontsize=13)
        ax.set_title(m, fontsize=14)
        # a single degenerate cell (e.g. hill_tail -> 1/eps) must not crush
        # the boxes: log-y when positive values span >2 orders of magnitude
        allv = np.concatenate([np.asarray(values[m][si], dtype=np.float64)
                               for si in range(n_sw)])
        allv = allv[np.isfinite(allv)]
        if allv.size and allv.min() > 0 and allv.max() / np.median(allv) > 100:
            ax.set_yscale('log')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=11)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {prefix}.{{pdf,png}}')
