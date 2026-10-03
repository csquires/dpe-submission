"""datagen diagnostic for pendulum trajectory ELDR experiment.

cells are k1_{i}_beta_{j}_seed_{s}.h5 under data_dir. the first axis is the
stratum, labelled by its realized K1 from alphas_chosen.yaml; the second is
beta, a fixed mixture weight read from config campaign.beta. nothing is
prescribed, so no panel compares a target against a realized value.

modes:
  default:  produce datagen_diagnostic.png, datagen_variance.png, the
            per-stratum data card, and the phase-space sample sheets.
  --pilot:  pilot mode with collapse diagnostics + go/no-go report.

panels (default mode):
  - realized K1 ladder bar plot (with error bars)
  - realized K2 per stratum
  - LDR histograms grid (per (k1_idx, beta_idx); seeds overlaid)
  - phase-space (theta, theta_dot) at t=0 and t=T
  - PCA of flat trajectories (first available cell)
  - per-cell KL/ELDR boxplot grid + summary table
  - data card with gate/replay/ckpt summaries
"""
import argparse
from glob import glob as glob_fn
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import _load_config
from ex.utils.diagnostic_primitives import collect_cells, plot_pca_panel


# standard name -> attr name in the per-cell hdf5 (step1_create_data schema).
# raw attrs are passed through as well, so hashes and the KL_* attrs are
# read under their own names.
KEY_MAP = {
    "alpha_chosen": "alpha_chosen",
    "k1_realized_flow": "K1_realized_flow",
    "k1_realized_flow_se": "K1_realized_flow_se",
    "k2_real": "K2_realized",
    "beta": "beta",
    "integrated_eldr": "integrated_eldr",
}

# per-cell boxplot panel titles; keys are the hardness metric names
PANEL_TITLES = {
    "integrated_eldr": "ground-truth ELDR",
    "k2_realized": r"$K_2 = \mathrm{KL}(p_* \| p_1)$",
    "KL_mix_O": r"$\mathrm{KL}(p_* \| p_0)$",
    "KL_O_E": r"$\mathrm{KL}(p_0 \| p_1)$",
    "KL_E_mix": r"$\mathrm{KL}(p_1 \| p_*)$",
    "mc_se": "ELDR MC standard error",
}


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="ex/semisynth/pendulum/config.yaml")
    p.add_argument("--skip-card", action="store_true",
                   help="skip the per-stratum data-card table + figure")
    p.add_argument("--skip-render", action="store_true",
                   help="skip the phase-space ground-truth sample sheet")
    p.add_argument("--pilot", action="store_true",
                   help="run 5-cell pilot mode: collapse diagnostics + go/no-go report")
    return p.parse_args(args)


def run_data_card(cells, k1_values, config):
    """per-stratum data card: dimensionality / multimodality / irregularity.

    metrics on the flattened (theta, theta_dot, action) trajectory samples of
    p* + true ldrs. emits data_card.{md,tex,png,pdf} into figures_dir.
    appends gate_summary, replay_summary, and flow_ckpt_hash info.
    """
    import json
    import h5py
    from ex.utils import data_card as dc

    names = ['twonn_id', 'part_ratio', 'gmm_modes', 'lip_q90', 'hill_tail']
    vals = {m: [[] for _ in k1_values] for m in names}
    for ki in range(len(k1_values)):
        for rec in cells.get((ki, 0), []):
            if 'samples_pstar' not in rec or 'true_ldrs' not in rec:
                continue
            X, ldr = rec['samples_pstar'], rec['true_ldrs']
            vals['twonn_id'][ki].append(dc.twonn_id(X))
            vals['part_ratio'][ki].append(dc.participation_ratio(X))
            vals['gmm_modes'][ki].append(dc.gmm_modes(X))
            vals['lip_q90'][ki].append(dc.lip_q(X, ldr))
            vals['hill_tail'][ki].append(dc.hill_tail(ldr))

    fig_dir = Path(config['figures_dir'])
    card_base = str(fig_dir / 'data_card')
    dc.write_card(card_base,
                  [f'K1={v:g}' for v in k1_values], vals,
                  title='pendulum data card (pstar trajectories) -- med [q1, q3] over seeds')
    dc.plot_metric_boxes(vals, k1_values, sweep_name='K1',
                         out_dir=str(fig_dir), prefix='data_card')

    # append gate_summary: reports carry policy_label / accepted / diagnostics
    import os as _os
    gate_dir = Path(_os.path.expandvars(config.get('gate', {}).get('out_dir', '')))
    gate_summaries = []
    if gate_dir.exists():
        for gate_json in sorted(gate_dir.glob('gate_*.json')):
            try:
                with open(gate_json) as f:
                    gate_data = json.load(f)
                label = gate_data.get('policy_label', '?')
                accepted = gate_data.get('accepted', False)
                diag = gate_data.get('diagnostics', {})
                gate_summaries.append(
                    f"  {label}: accepted={accepted}, "
                    f"pit_ks_stat={diag.get('pit_ks_statistic')}, "
                    f"nll_gap={gate_data.get('nll_details', {}).get('gap')}")
            except Exception:
                pass

    # replay + checkpoint provenance, one line per stratum from its first
    # cell: the O-policy replay (replay_{rl_hash}.h5: pair count + phase
    # balance) and the E/O flow hashes (the E hash is shared and repeats)
    replay_summaries = []
    ckpt_hashes = []
    replay_dir = Path(_os.path.expandvars(config.get('rl_runs', {}).get('cache_dir', '')))
    for (ki, _), recs in sorted(cells.items()):
        attrs = recs[0].get('attrs', {}) if recs else {}
        if 'flow_ckpt_hash_O' not in attrs:
            continue
        label = f'K1={k1_values[ki]:g}'
        ckpt_hashes.append(
            f"{label}: flow_E={attrs.get('flow_ckpt_hash_E')} "
            f"flow_O={attrs['flow_ckpt_hash_O']} rl_O={attrs.get('rl_ckpt_hash_O')}")
        replay_h5 = replay_dir / f"replay_{attrs.get('rl_ckpt_hash_O')}.h5"
        if not replay_h5.exists():
            continue
        try:
            with h5py.File(replay_h5, 'r') as f:
                n_pairs = int(f['states'].shape[0])
                phase = np.bincount(f['phase'][:].astype(np.int64), minlength=3)
            replay_summaries.append(
                f"  {label}: n_pairs={n_pairs}, phase_balance=(early={phase[0]}, "
                f"mid={phase[1]}, late={phase[2]})")
        except Exception:
            pass

    # append all summaries to data_card markdown file
    card_md = card_base + '.md'
    if Path(card_md).exists():
        with open(card_md, 'a') as f:
            if gate_summaries:
                f.write('\n## Gate Summary\n')
                for summary in gate_summaries:
                    f.write(summary + '\n')
            if replay_summaries:
                f.write('\n## Replay Summary\n')
                for summary in replay_summaries:
                    f.write(summary + '\n')
            if ckpt_hashes:
                f.write('\n## Flow Checkpoint Hashes\n')
                for h in ckpt_hashes:
                    f.write(f'  {h}\n')


def run_pilot_report(cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                     config: Dict[str, Any]) -> None:
    """pilot diagnostic: collapse detection + go/no-go decision.

    pre-registered thresholds (frozen for reproducibility):
      PILOT_MIN_LDR_VAR: floor on realized LDR variance
      PILOT_MAX_ACTION_MODES: upper bound on action mode count (t=0)
      PILOT_KS_GATE_THRESHOLD: KS stat gate threshold

    procedure:
      1. identify the pilot cells from config['pilot_cell_keys']
      2. per cell: ldr histogram + variance, action mode count via smoothed sign changes
      3. load gate pass/fail from gate reports
      4. go/no-go: (ldr_var >= MIN) AND (mode_count <= MAX) AND (gate_pass)
      5. write pilot_report.md to figures_dir
    """
    import json
    import os
    from scipy import signal

    # ========== pre-registered constants block (frozen for reproducibility) ==========
    PILOT_MIN_LDR_VAR = 0.3  # floor on realized LDR variance
    PILOT_MAX_ACTION_MODES = 3  # upper bound on action mode count (unimodal + 2 neighbors)
    PILOT_KS_GATE_THRESHOLD = 0.05  # KS stat from gate report
    # ==================================================================================

    pilot_cell_keys = config.get('pilot_cell_keys', [])
    if not pilot_cell_keys:
        print("pilot_cell_keys not defined in config; skipping pilot report")
        return

    # collect pilot diagnostics
    pilot_rows = []
    n_pass = 0

    for cell_key in pilot_cell_keys:
        ki, bi = cell_key
        recs = cells.get((ki, bi), [])
        if not recs:
            pilot_rows.append((cell_key, np.nan, None, False, "FAIL (no data)"))
            continue

        rec = recs[0]  # use first seed
        attrs = rec.get('attrs', {})

        # ldr histogram + variance
        ldr_var = None
        if 'true_ldrs' in rec:
            ldrs = rec['true_ldrs']
            if len(ldrs) > 1:
                ldr_var = float(np.var(ldrs))
        if ldr_var is None:
            ldr_var = np.nan

        # action mode count: extract t=0 actions, smooth, count sign changes in derivative
        mode_count = None
        if 'samples_p0' in rec and 'samples_p1' in rec:
            T = int(config.get('trajectory', {}).get('T', 5))
            try:
                # reshape to [N, T+1, 3] and extract action at t=0
                p0_actions = rec['samples_p0'].reshape(-1, T + 1, 3)[:, 0, 2]  # [N] actions at t=0
                p1_actions = rec['samples_p1'].reshape(-1, T + 1, 3)[:, 0, 2]

                # density-peak count: smoothed histogram + prominent peaks
                # (savgol on sorted samples counts quantile wiggles, not modes)
                if len(p0_actions) > 5:
                    from scipy.ndimage import gaussian_filter1d
                    hist, _ = np.histogram(
                        np.concatenate([p0_actions, p1_actions]),
                        bins=60, range=(-2.0, 2.0))
                    smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
                    peaks, _ = signal.find_peaks(
                        smooth, prominence=0.05 * smooth.max())
                    mode_count = int(len(peaks))
                else:
                    mode_count = 1
            except Exception:
                mode_count = None

        if mode_count is None:
            mode_count = 0

        # gate pass/fail: reports are content-addressed by ckpt hash and the
        # cell attrs carry both hashes -> require both E and O accepted
        gate_pass = False
        gate_dir = Path(os.path.expandvars(config.get('gate', {}).get('out_dir', '')))
        h_e = attrs.get('flow_ckpt_hash_E')
        h_o = attrs.get('flow_ckpt_hash_O')
        if gate_dir.exists() and h_e and h_o:
            def _accepted(h):
                fp = gate_dir / f"gate_{h}.json"
                if not fp.exists():
                    return False
                try:
                    with open(fp) as f:
                        return bool(json.load(f).get("accepted", False))
                except Exception:
                    return False
            gate_pass = _accepted(h_e) and _accepted(h_o)

        # compute go/no-go
        go_no_go = (ldr_var >= PILOT_MIN_LDR_VAR) and (mode_count <= PILOT_MAX_ACTION_MODES) and gate_pass
        status = "PASS" if go_no_go else "FAIL"
        if go_no_go:
            n_pass += 1

        pilot_rows.append((cell_key, ldr_var, mode_count, gate_pass, status))

    # write pilot_report.md
    fig_dir = Path(config['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_path = fig_dir / 'pilot_report.md'

    with open(report_path, 'w') as f:
        f.write(f'# {len(pilot_rows)}-cell pilot report (pre-registered thresholds; cells indexed by stratum label)\n\n')
        f.write('| cell_key | ldr_var | mode_count | gate_pass | go/no-go |\n')
        f.write('|----------|---------|-----------|-----------|----------|\n')
        for cell_key, ldr_var, mode_count, gate_pass, status in pilot_rows:
            ldr_str = f"{ldr_var:.3f}" if not np.isnan(ldr_var) else "NaN"
            mode_str = str(mode_count) if mode_count is not None else "?"
            gate_str = "True" if gate_pass else "False"
            f.write(f'| {cell_key} | {ldr_str} | {mode_str} | {gate_str} | {status} |\n')
        f.write('\n## Summary\n')
        f.write(f'{n_pass}/{len(pilot_rows)} ready; recommend: {"proceed with caution" if n_pass >= 3 else "investigate further"}\n')
        f.write(f'\n## Pre-registered Constants\n')
        f.write(f'- PILOT_MIN_LDR_VAR = {PILOT_MIN_LDR_VAR}\n')
        f.write(f'- PILOT_MAX_ACTION_MODES = {PILOT_MAX_ACTION_MODES}\n')
        f.write(f'- PILOT_KS_GATE_THRESHOLD = {PILOT_KS_GATE_THRESHOLD}\n')

    print(f"saved {report_path}")


def run_sample_render(cells, k1_values, config, n_show=10, n_seeds=2):
    """ground-truth rendering: phase-space (theta, theta_dot) rollouts.

    n_show sampled trajectories per distribution for n_seeds seeds per K1;
    samples reshape to (T+1, 3) = per-step (theta, theta_dot, action).
    """
    dists = [('samples_p0', r'$p_0$ ($\pi_O$)'),
             ('samples_p1', r'$p_1$ ($\pi_E$)'),
             ('samples_pstar', r'$p_*$')]

    def wrap_breaks(t):
        """insert nan rows where theta jumps past +-pi so segments don't bridge the wrap"""
        jump = np.where(np.abs(np.diff(t[:, 0])) > np.pi)[0]
        return np.insert(t, jump + 1, np.nan, axis=0) if jump.size else t

    rng = np.random.default_rng(0)
    out = Path(config['figures_dir'])
    # one file per K1 so figures stay paper-sized and composable
    for ki, k1 in enumerate(k1_values):
        recs = cells.get((ki, 0), [])[:n_seeds]
        if not recs:
            continue
        fig, axes = plt.subplots(len(recs), 3,
                                 figsize=(3.4 * 3, 2.8 * len(recs)),
                                 sharex=True, sharey=True, squeeze=False)
        for r, rec in enumerate(recs):
            for ci, (dk, lab) in enumerate(dists):
                ax = axes[r, ci]
                X = rec[dk]
                idx = rng.choice(X.shape[0], n_show, replace=False)
                for t in X[idx].reshape(n_show, -1, 3):
                    tb = wrap_breaks(t)
                    ax.plot(tb[:, 0], tb[:, 1], '-o', markersize=2.5,
                            linewidth=0.9, alpha=0.7)
                    ax.plot(t[0, 0], t[0, 1], 'k.', markersize=6)
                ax.set_title(f'{lab}  seed={rec["seed"]}', fontsize=11)
                ax.grid(True, alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel(r'$\theta$')
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$\dot\theta$')
        fig.tight_layout()
        tag = f'{k1:g}'.replace('.', 'p')
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'datagen_samples_k1_{tag}.{ext}', dpi=150,
                        bbox_inches='tight')
        plt.close(fig)
        print(f'saved datagen_samples_k1_{tag}.{{pdf,png}}')


# ----------------------------------------------------------------------
# config / path resolution
# ----------------------------------------------------------------------

def stratum_axes(config: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """(k1_values, beta_values) that label the cell axes.

    k1_values: realized K1 per stratum from alphas_chosen.yaml, in
    stratum_label order (= k1_idx order), rounded to one decimal like step4.
    beta_values: the single fixed mixture weight from config campaign.beta.
    """
    from ex.utils.realized_kl_table import load_alphas_chosen
    strata = sorted(load_alphas_chosen(config["data_dir"])["strata"],
                    key=lambda s: s["stratum_label"])
    k1_values = [round(float(s["K1_realized_flow"]), 1) for s in strata]
    return k1_values, [float(config["campaign"]["beta"])]


def enumerate_cell_paths(config: Dict[str, Any]
                         ) -> Dict[Tuple[int, int], List[Tuple[int, str]]]:
    """walk data_dir for k1_{i}_beta_{j}_seed_{s}.h5; group by (k1_idx, beta_idx).

    returns dict[(k1_idx, beta_idx)] -> list of (seed, path), seeds sorted;
    only existing files included.
    """
    data_dir = Path(config["data_dir"])
    out: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for path in sorted(glob_fn(str(data_dir / "k1_*_beta_*_seed_*.h5"))):
        parts = Path(path).stem.split('_')
        try:
            key = (int(parts[1]), int(parts[3]))
            seed = int(parts[5])
        except (IndexError, ValueError):
            continue
        out.setdefault(key, []).append((seed, path))
    for key in out:
        out[key].sort(key=lambda x: x[0])
    return out


# ----------------------------------------------------------------------
# per-cell aggregate plots
# ----------------------------------------------------------------------

def plot_k1_ladder_bar(ax, config: Dict[str, Any]) -> None:
    """bar plot of realized K1 ladder: one bar per chosen stratum with error bars.

    reads the canonical alphas_chosen.yaml (ladder/strata) for alphas, realized K1, SE.
    highlights chosen strata from the selection process.
    """
    # read the canonical alphas_chosen.yaml (ladder field carries the
    # measured entries; strata are the chosen ones)
    try:
        from ex.utils.realized_kl_table import load_alphas_chosen
        doc = load_alphas_chosen(config["data_dir"])
        chosen = doc.get("ladder", []) or [
            {"alpha": s["alpha"], "kl_hat": s["K1_realized_flow"],
             "kl_se": s["K1_se"], "stratum_label": s["stratum_label"]}
            for s in doc["strata"]]
    except Exception as e:
        ax.text(0.5, 0.5, f"alphas_chosen.yaml unavailable: {e}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_visible(False)
        return

    if not chosen:
        ax.text(0.5, 0.5, "no measured strata in alphas_chosen.yaml",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_visible(False)
        return

    # extract data from chosen strata
    alphas = np.array([e["alpha"] for e in chosen])
    k1_hats = np.array([e.get("kl_hat", 0) for e in chosen])
    k1_ses = np.array([e.get("kl_se", 0) for e in chosen])
    stratum_labels = [e.get("stratum_label", i) for i, e in enumerate(chosen)]

    # bar plot with error bars
    x_pos = np.arange(len(alphas))
    colors = ["tab:orange" if i == sl else "tab:blue" for i, sl in enumerate(stratum_labels)]
    # all are chosen, so highlight them uniformly; adjust if subset highlighting needed
    ax.bar(x_pos, k1_hats, yerr=k1_ses, capsize=5, alpha=0.7, color="tab:orange", edgecolor="black", linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([rf"$\alpha={a:g}$" for a in alphas], fontsize=9)
    ax.set_ylabel(r"$K_1$ realized (±SE)")
    ax.set_title(r"Realized $K_1$ ladder (selected strata)")


def plot_k1_vs_k2_realized(ax,
                           cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                           k1_values: List[float],
                           beta_values: List[float]) -> None:
    r"""$K_1$ realized vs $K_2$ realized; one line per $\beta$ value.

    each marker is the median across seeds at one (K1, $\beta$) cell. with
    singleton $\beta$ this is a single monotone line; with swept $\beta$ this
    is one line per $\beta$, color-coded.
    """
    n2 = len(beta_values)
    by_beta: Dict[int, List[Tuple[float, float]]] = {}
    for (ai, bi), recs in cells.items():
        vals = [r["attrs"]["k2_real"] for r in recs
                if "k2_real" in r["attrs"]]
        if not vals:
            continue
        by_beta.setdefault(bi, []).append(
            (float(k1_values[ai]), float(np.median(vals))))
    for bi, pts in sorted(by_beta.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        col = plt.cm.plasma(bi / max(1, n2 - 1))
        ax.plot(xs, ys, "-o", color=col, markersize=5,
                label=rf"$\beta$={beta_values[bi]}")
    ax.set_xlabel(r"$K_1$ realized")
    ax.set_ylabel("$K_2$ realized\n(median over seeds)")
    ax.set_title(r"$K_1$ vs $K_2$ realized (one line per $\beta$)")
    if n2 > 1:
        ax.legend(fontsize=7)


def plot_ldr_histograms_grid(
    fig, gs_slice,
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
    k1_values: List[float], beta_values: List[float],
    extract_ldrs: Callable[[Dict[str, Any]], Optional[np.ndarray]],
    title_prefix: str = "log p0 / p1",
) -> None:
    """grid of ldr histograms, axes_ij = (k1_idx, beta_idx). seeds overlaid.

    args:
      fig: matplotlib figure.
      gs_slice: a SubplotSpec sliced to the desired sub-region.
      cells: per-cell records.
      k1_values, beta_values: sweep axis values (used for titles only).
      extract_ldrs: callable taking a single cell record -> ldrs np.ndarray.
                    None return is skipped.
      title_prefix: title prefix for each subplot.
    """
    n1 = len(k1_values)
    n2 = len(beta_values)
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        n1, n2, subplot_spec=gs_slice, hspace=0.5, wspace=0.4)
    for ai in range(n1):
        for bi in range(n2):
            ax = fig.add_subplot(sub_gs[ai, bi])
            recs = cells.get((ai, bi), [])
            for r in recs:
                ldrs = extract_ldrs(r)
                if ldrs is None or len(ldrs) == 0:
                    continue
                ax.hist(ldrs, bins=50, density=True, alpha=0.25,
                        color="tab:blue")
            ax.axvline(0, color="black", linestyle="--", linewidth=0.4)
            ax.set_yscale("log")
            ax.set_title(
                rf"$K_1$={k1_values[ai]}, $\beta$={beta_values[bi]}",
                fontsize=7)
            ax.tick_params(labelsize=6)
    pos = gs_slice.get_position(fig)
    fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.02,
             f"{title_prefix} histograms (rows: $K_1$, cols: $\\beta$)",
             ha="center", va="bottom", fontsize=10)


# ----------------------------------------------------------------------
# hardness aggregation
# ----------------------------------------------------------------------

def compute_hardness(
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
    k1_values: List[float], beta_values: List[float],
    extra_metrics: Optional[Dict[str, Callable[[Dict[str, Any]], float]]] = None,
) -> Dict[str, np.ndarray]:
    """compute per-cell hardness statistics.

    standard metrics: integrated_eldr, k2_realized. nothing is prescribed
    (alpha is set, K1 is measured, beta is fixed), so there is no target to
    subtract from and no fidelity metric.

    args:
      cells: per-cell records.
      k1_values, beta_values: define the (n1, n2) shape.
      extra_metrics: dict[name -> callable(cell_record) -> float], experiment-specific.

    returns:
      dict[name -> array of shape [n1, n2, max_seeds]] padded with NaN where missing.
    """
    n1 = len(k1_values)
    n2 = len(beta_values)
    max_seeds = max((len(v) for v in cells.values()), default=0)

    standard = {
        "integrated_eldr": lambda r: r["attrs"].get("integrated_eldr", np.nan),
        "k2_realized": lambda r: r["attrs"].get("k2_real", np.nan),
    }
    metrics = {**standard, **(extra_metrics or {})}

    out = {name: np.full((n1, n2, max_seeds), np.nan) for name in metrics}
    for (ai, bi), recs in cells.items():
        for si, r in enumerate(recs):
            for name, fn in metrics.items():
                try:
                    out[name][ai, bi, si] = float(fn(r))
                except Exception:
                    out[name][ai, bi, si] = np.nan
    return out


def print_hardness_table(hardness: Dict[str, np.ndarray],
                         k1_values: List[float],
                         beta_values: List[float]) -> None:
    """print median/iqr/mean/std summary across seeds, per cell, per metric."""
    print("\n" + "=" * 80)
    print("HARDNESS / VARIANCE SUMMARY (per (k1_idx, beta_idx), across seeds)")
    print("=" * 80)
    for name, arr in hardness.items():
        print(f"\n--- {name} ---")
        print(f"{'k1':>6} {'beta':>6} {'median':>9} {'IQR':>8} "
              f"{'mean':>9} {'std':>8} {'n_seeds':>8}")
        for ai, k1 in enumerate(k1_values):
            for bi, b in enumerate(beta_values):
                row = arr[ai, bi]
                row = row[~np.isnan(row)]
                if len(row) == 0:
                    continue
                q25, q50, q75 = np.percentile(row, [25, 50, 75])
                print(f"{k1:>6.2f} {b:>6.2f} {q50:>9.3f} "
                      f"{q75 - q25:>8.3f} {np.mean(row):>9.3f} "
                      f"{np.std(row):>8.3f} {len(row):>8d}")


def plot_hardness_boxplots(hardness: Dict[str, np.ndarray],
                           k1_values: List[float],
                           beta_values: List[float],
                           fig_path: str) -> None:
    """boxplot grid: one panel per metric, boxes per k1_idx (collapsing beta+seed).

    each box aggregates all (beta_idx, seed) at that k1_idx. richer view
    is the print_hardness_table output.
    """
    del beta_values  # used by table, not by boxplot
    names = list(hardness.keys())
    n = len(names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    # rc context stays open through savefig so draw-time tick labels inherit the
    # shared bold/2x box-plot text spec; panels are sized to match.
    from ex.utils.plot_style import box_style
    with box_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.0 * nrows),
                                 squeeze=False)
        n1 = len(k1_values)
        for i, name in enumerate(names):
            ax = axes[i // ncols, i % ncols]
            arr = hardness[name]
            per_k1 = []
            for ai in range(n1):
                row = arr[ai].ravel()
                row = row[~np.isnan(row)]
                per_k1.append(row if len(row) > 0 else np.array([np.nan]))
            bp = ax.boxplot(per_k1, tick_labels=[f"{k:g}" for k in k1_values],
                            patch_artist=True, showfliers=True,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor("tab:blue")
                patch.set_alpha(0.4)
            ax.set_xlabel(r"$K_1$ realized")
            ax.set_ylabel(PANEL_TITLES.get(name, name))
            ax.set_title(PANEL_TITLES.get(name, name))
        for i in range(n, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)
        fig.tight_layout(pad=0.3)
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {fig_path}")


# ----------------------------------------------------------------------
# pendulum-specific extraction + panels
# ----------------------------------------------------------------------

def extract_pendulum_ldrs(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    r"""log $p_O$ - log $p_E$ at pstar samples = log($p_0/p_1$)."""
    log_p = rec.get("log_p_pstar")
    if log_p is None:
        return None
    return log_p[:, 1] - log_p[:, 2]


def plot_phase_space(ax, samples: np.ndarray, T: int, title: str,
                     t_idx: int) -> None:
    r"""scatter ($\theta$, $\dot\theta$) marginal at timestep t_idx.

    args:
      ax: axes.
      samples: [N, (T+1)*3] flat, columns = ($\theta$, $\dot\theta$, action) per step.
      T: trajectory length T (so samples reshape to [N, T+1, 3]).
      title: subplot title.
      t_idx: which timestep index in [0, T] to plot.
    """
    N = samples.shape[0]
    arr = samples.reshape(N, T + 1, 3)
    theta = arr[:, t_idx, 0]
    theta_dot = arr[:, t_idx, 1]
    ax.scatter(theta, theta_dot, s=3, alpha=0.3, color="tab:blue")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot\theta$")
    ax.set_title(title)


def plot_phase_space_grid(fig, gs_slice,
                          cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                          k1_values: List[float],
                          beta_values: List[float],
                          T: int) -> None:
    """grid of phase-space scatters at t=0 (left) and t=T (right) for pstar samples.

    one row per (k1_idx, beta_idx) cell; first seed only.
    """
    items = sorted(cells.keys())
    n_rows = len(items)
    if n_rows == 0:
        return
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, 2, subplot_spec=gs_slice, hspace=0.5, wspace=0.3)
    for ri, (ai, bi) in enumerate(items):
        rec = cells[(ai, bi)][0]
        if "samples_pstar" not in rec:
            continue
        ax0 = fig.add_subplot(sub_gs[ri, 0])
        ax_T = fig.add_subplot(sub_gs[ri, 1])
        # column titles on top row only; row info as left-col ylabel
        col_t0 = "initial (t=0)" if ri == 0 else ""
        col_tT = f"terminal (t={T})" if ri == 0 else ""
        plot_phase_space(ax0, rec["samples_pstar"], T, col_t0, t_idx=0)
        plot_phase_space(ax_T, rec["samples_pstar"], T, col_tT, t_idx=T)
        ax0.set_ylabel(
            rf"$K_1$={k1_values[ai]}, $\beta$={beta_values[bi]}"
            "\n" r"$\dot\theta$", fontsize=8)


# ----------------------------------------------------------------------
# lightweight figure assembly + main
# ----------------------------------------------------------------------

def plot_lightweight_figure(cells: Dict[Tuple[int, int], List[Dict[str, Any]]],
                            config: Dict[str, Any],
                            k1_values: List[float],
                            beta_values: List[float]) -> None:
    """assemble lightweight figure to figures_dir/datagen_diagnostic.png."""
    T = int(config["trajectory"]["T"])
    n1 = len(k1_values)
    n2 = len(beta_values)
    n_cells = len(cells)

    # rows: (0) summary 1x2, (1) ldr histogram block n1*n2,
    #       (2) phase space block n_cells rows, (3) pca scatter 1 row
    fig = plt.figure(figsize=(4 * max(n1, n2, 4), 4 + 2 * n1 + 2 * n_cells + 4))
    gs = gridspec.GridSpec(4, 1, figure=fig,
                           height_ratios=[1, n1 * 1.2, max(1, n_cells), 2],
                           hspace=0.5)

    sub0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                             wspace=0.4)
    plot_k1_ladder_bar(fig.add_subplot(sub0[0, 0]), config)
    plot_k1_vs_k2_realized(fig.add_subplot(sub0[0, 1]), cells,
                           k1_values, beta_values)

    plot_ldr_histograms_grid(fig, gs[1], cells, k1_values, beta_values,
                             extract_pendulum_ldrs,
                             title_prefix=r"$\log p_O - \log p_E$ at $p^*$")

    plot_phase_space_grid(fig, gs[2], cells, k1_values, beta_values, T)

    pca_cell = next(iter(cells.values()), None)
    if pca_cell is not None and "samples_pstar" in pca_cell[0]:
        sub3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[3])
        ax_pca = fig.add_subplot(sub3[0, 0])
        rec = pca_cell[0]
        plot_pca_panel(ax_pca, rec["samples_pstar"], rec["samples_p0"],
                       rec["samples_p1"],
                       title=f"PCA, first available cell, seed={rec['seed']}")

    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "datagen_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    args = parse_args()
    config = _load_config(args.config)

    paths_by_idx = enumerate_cell_paths(config)
    if not paths_by_idx:
        print(f"no per-cell HDF5 files found under {config['data_dir']}. "
              f"run step1_create_data.py first.")
        return

    cells = collect_cells(paths_by_idx, KEY_MAP)

    if args.pilot:
        run_pilot_report(cells, config)
        return

    # default flow: full diagnostics; axes are labelled by realized K1 + beta
    k1_values, beta_values = stratum_axes(config)

    plot_lightweight_figure(cells, config, k1_values, beta_values)

    # panel order = row-major in datagen_variance: eldr and its two halves,
    # then the endpoint KLs and the mc error
    hardness = compute_hardness(
        cells,
        k1_values,
        beta_values,
        extra_metrics={
            "KL_mix_O": lambda r: r["attrs"].get("KL_mix_O", np.nan),
            "KL_O_E": lambda r: r["attrs"].get("KL_O_E", np.nan),
            "KL_E_mix": lambda r: r["attrs"].get("KL_E_mix", np.nan),
            "mc_se": lambda r: r["attrs"].get("mc_se", np.nan),
        },
    )
    print_hardness_table(hardness, k1_values, beta_values)
    plot_hardness_boxplots(hardness,
                           k1_values,
                           beta_values,
                           str(Path(config["figures_dir"]) / "datagen_variance.png"))

    if not args.skip_card:
        run_data_card(cells, k1_values, config)
    if not args.skip_render:
        run_sample_render(cells, k1_values, config)


if __name__ == "__main__":
    main()
