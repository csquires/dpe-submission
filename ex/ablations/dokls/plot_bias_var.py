"""figures and tables for the dokls (two-leg) vs plug-in bias-variance split.

reads processed_results/bias_var.h5 (written by bias_var.py) and writes to figures_dir:
    dokls_bv_kl_low.{pdf,png}   kl 0.3, 1, 3
    dokls_bv_kl_high.{pdf,png}  kl 9, 18, 36, 54
    dokls_bv_table.{md,tex}     dokls minus plug-in mse, split into its bias^2 part
                                and its variance part (units of the plug-in mse), one
                                section per (p*, N*), one row per family
    dokls_bv_summary.{md,tex}   counts of higher / lower dokls mse (and of clear
                                differences) and of variance-led gaps, per kl range

figure layout (rows = kl levels):
    columns: 6 families for q0, then 6 for q1 (width prop. to bar count)
    each family: rmse tier (own log10 axis, dots with 95% boot ci, all routes) over
    an mse tier (plug-in and dokls bars on one scale per N*, the larger mse = 1,
    each split into bias^2 and variance); x = N* subgroups, routes side by side
    marks: triangle = rmse above 30x the family's median plug-in/dokls rmse (off
    scale). cells with fewer than 10 finite instances are left out (only the nwj
    legs at q1 lose instances; the plug-in and dokls rows keep all 10).
the 10.8 in canvas prints at \\textwidth (6.75 in, 0.625x), so fonts 13-15 -> 8-9 pt.
colors: validated categorical slots 1-4 in bar order (plug-in, dokls, nwj, dv).

usage: python -m ex.ablations.dokls.plot_bias_var
"""
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import numpy as np

from ex.ablations.dokls.variants import resolve
from ex.utils.plot_style import display_name

INK, INK2, GRID, SPINE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
COLOR = {'plugin': '#2a78d6', 'two_leg': '#eb6834', 'nwj': '#1baf7a', 'dv': '#eda100'}
LABEL = {'plugin': 'plug-in', 'two_leg': 'DoKLs', 'nwj': 'DoKLs NWJ', 'dv': 'DoKLs DV'}
FS = dict(ms=4.6, tick=13, ns=13, fam=13.5, lab=13, row=14.5, leg=13, note=12)
BAR, GAP_N, CAP = 0.86, 0.45, 30.0
BANDS = {'low': [0, 1, 2], 'high': [3, 4, 5, 6]}
PAIR = ('plugin', 'two_leg')


def load(path):
    """bias_var.h5 -> dict of arrays + rows [(route key, method, family)] + axes."""
    with h5py.File(path, 'r') as f:
        d = {k: f[k][()] for k in ('bias', 'var', 'mse', 'n', 'rmse', 'rmse_lo',
                                   'rmse_hi', 'dbias2', 'dvar', 'dmse_lo', 'dmse_hi',
                                   'dbias2_lo', 'dvar_lo')}
        fams = [s.decode() for s in f['families'][()]]
        rows = []
        for s, fam in zip(f['rows'][()], f['row_family'][()]):
            route, m = s.decode().split('/')
            key = 'nwj' if m.endswith('_NWJ') else 'dv' if m.endswith('_DV') else route
            rows.append((key, m, fam.decode()))
        d.update(rows=rows, families=fams, nstar=list(f.attrs['nstar']),
                 kl=list(f.attrs['kl']))
    return d


def tint(hexc, a=0.38):
    """blend a hex color toward white; a = weight of the color."""
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(a * c + (1 - a))


def xpos(d, fam):
    """bar x per (nstar index, row index), subgroup centers, x limits for one family."""
    idx = [r for r, (_, _, f) in enumerate(d['rows']) if f == fam]
    pos, cen = {}, []
    for s in range(len(d['nstar'])):
        xs = s * (len(idx) + GAP_N) + np.arange(len(idx))
        pos.update({(s, r): x for r, x in zip(idx, xs)})
        cen.append(xs.mean())
    return pos, cen, (-0.6, xs[-1] + 0.6)


def chrome(ax):
    """hairline y grid, muted spines, short ticks."""
    ax.grid(True, axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=INK2, labelsize=FS['tick'], length=2, pad=1.5)


def cell(top, bot, d, fam, p, k, first_col, last_row):
    """one family for (p*, kl): rmse dots + ci over the paired bias^2/variance bars."""
    pos, cen, xl = xpos(d, fam)
    xt = top.get_xaxis_transform()
    base = [d['rmse'][r, p, s, k] for (s, r) in pos if d['rows'][r][0] in PAIR]
    cap = CAP * np.median(base)
    vals = []
    for (s, r), x in pos.items():
        key = d['rows'][r][0]
        rm = d['rmse'][r, p, s, k]
        if d['n'][r, p, s, k] < 10:
            continue
        if rm > cap:
            top.plot(x, 1.0, marker='^', color=COLOR[key], ms=FS['ms'] + 1.5,
                     mec='white', mew=0.6, transform=xt, clip_on=False, zorder=4)
            continue
        lo, hi = d['rmse_lo'][r, p, s, k], d['rmse_hi'][r, p, s, k]
        vals += [lo, min(hi, cap)]
        top.errorbar(x, rm, yerr=[[rm - lo], [hi - rm]], fmt='o', ms=FS['ms'],
                     color=COLOR[key], mec='white', mew=0.6, elinewidth=1.0,
                     capsize=0, zorder=3)
    for s in range(len(d['nstar'])):
        pair = [r for (ss, r) in pos if ss == s and d['rows'][r][0] in PAIR]
        scale = max(d['mse'][r, p, s, k] for r in pair)
        for r in pair:
            key = d['rows'][r][0]
            b2 = d['bias'][r, p, s, k] ** 2 / scale
            bot.bar(pos[(s, r)], b2, BAR, color=COLOR[key], edgecolor='white',
                    linewidth=0.5)
            bot.bar(pos[(s, r)], d['var'][r, p, s, k] / scale, BAR, bottom=b2,
                    color=tint(COLOR[key]), edgecolor='white', linewidth=0.5)
    top.set_yscale('log')
    lo, hi = 10 ** np.floor(np.log10(min(vals))), 10 ** np.ceil(np.log10(max(vals)))
    top.set_ylim(lo / 1.6, hi * (hi / lo) ** 0.3)
    top.yaxis.set_major_locator(mtick.LogLocator(base=10, numticks=4))
    top.yaxis.set_major_formatter(mtick.LogFormatterExponent())
    top.yaxis.set_minor_formatter(mtick.NullFormatter())
    for ax in (top, bot):
        chrome(ax)
        ax.set_xlim(*xl)
    top.tick_params(axis='x', length=0, labelbottom=False)
    bot.set_ylim(0, 1)
    bot.set_yticks([0, 0.5, 1])
    bot.set_xticks(cen)
    if last_row:
        bot.set_xticklabels([f'{ns // 1024}' for ns in d['nstar']], fontsize=FS['ns'])
    else:
        bot.tick_params(axis='x', labelbottom=False, length=0)
    if first_col:
        bot.set_yticklabels(['0', '.5', '1'])
    else:
        bot.tick_params(axis='y', labelleft=False)


def figure(d, ks, path):
    """rows = kl levels ks; columns = families x {q0, q1}; shared legend below."""
    row_h, head, foot = 2.75, 0.85, 1.55
    h = row_h * len(ks) + head + foot
    fig = plt.figure(figsize=(10.8, h))
    outer = fig.add_gridspec(len(ks), 2, wspace=0.1, hspace=0.33, left=0.078,
                             right=0.984, top=1 - head / h, bottom=foot / h)
    fams = d['families']
    widths = [3 * sum(f == x for *_, x in d['rows']) + 2 * GAP_N + 0.2 for f in fams]
    lead, row0 = {}, {}
    for i, k in enumerate(ks):
        for j, p in enumerate((0, 1)):
            sub = outer[i, j].subgridspec(2, len(fams), width_ratios=widths,
                                          height_ratios=(2, 1.05), wspace=0.5,
                                          hspace=0.08)
            share_ax = None
            for c, fam in enumerate(fams):
                top = fig.add_subplot(sub[0, c])
                bot = fig.add_subplot(sub[1, c], sharey=share_ax)
                share_ax = share_ax or bot
                cell(top, bot, d, fam, p, k, c == 0, i == len(ks) - 1)
                if i == 0:
                    top.set_title(display_name(fam), fontsize=FS['fam'], color=INK, pad=12)
                    row0.setdefault(j, []).append(top)
                if c == 0 and j == 0:
                    top.set_ylabel(r'log$_{10}$ RMSE', color=INK2, fontsize=FS['lab'])
                    bot.set_ylabel('MSE\n(scaled)', color=INK2, fontsize=FS['lab'])
                    lead[i] = top
    for i, ax in lead.items():
        b = ax.get_position()
        y = 1 - 0.1 / h if i == 0 else b.y1 + 0.1 / h
        fig.text(0.005, y, rf'KL$(p_0\,\|\,p_1) = {d["kl"][ks[i]]:g}$', fontsize=FS['row'],
                 color=INK, ha='left', va='top' if i == 0 else 'bottom')
    for j, axs in row0.items():
        x0, x1 = axs[0].get_position().x0, axs[-1].get_position().x1
        fig.text((x0 + x1) / 2, 1 - 0.1 / h, rf'$p_* = q_{j}$', fontsize=FS['row'] + 0.5,
                 color=INK, ha='center', va='top')
    handles = [Line2D([], [], marker='o', ls='', color=COLOR[c], mec='white', ms=7)
               for c in COLOR] + [Patch(color='#5f5e5a'), Patch(color=tint('#5f5e5a')),
                                  Line2D([], [], marker='^', ls='', color=INK2, ms=7)]
    labels = [LABEL[c] for c in COLOR] + ['bias$^2$', 'variance', 'off scale']
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               fontsize=FS['leg'], bbox_to_anchor=(0.53, 0.34 / h), handletextpad=0.3,
               columnspacing=1.6)
    fig.text(0.53, 0.06 / h, r'x: $N_*$ ($\times$1024) in each panel;  bottom tier: '
             r'plug-in and DoKLs MSE on one scale (larger = 1);  own RMSE axis per family',
             ha='center', va='bottom', fontsize=FS['note'], color=INK2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', dpi=150, facecolor='white')
    plt.close(fig)
    print(f'  saved {os.path.basename(path)}.{{pdf,png}}')


def fmt_diff(db, dv, mse):
    """'bias^2 part / variance part' of an mse difference, in units of mse (3 sig figs)."""
    return f'{db / mse:+.3g} / {dv / mse:+.3g}'


def table(d, stem):
    """{stem}.md and {stem}.tex: one section per (p*, N*), one row per family."""
    header = ['Family'] + [f'KL={k:g}' for k in d['kl']]
    rows = [r for r, (key, m, fam) in enumerate(d['rows']) if key == 'plugin']
    md, tex = [], ['% generated by ex/ablations/dokls/plot_bias_var.py from bias_var.h5']
    for p in (0, 1):
        for s, ns in enumerate(d['nstar']):
            body = [[display_name(d['rows'][r][2])] + [
                fmt_diff(d['dbias2'][f, p, s, k], d['dvar'][f, p, s, k], d['mse'][r, p, s, k])
                for k in range(len(d['kl']))] for f, r in enumerate(rows)]
            title = ('MSE difference, DoKLs minus plug-in, split into its bias$^2$ part / its '
                     'variance part, in units of the plug-in MSE (the two parts add up to the '
                     f'relative difference) -- $p_* = q_{p}$, $N = 8192$, $N_* = {ns}$')
            md += [f'## {title}', '', '| ' + ' | '.join(header) + ' |',
                   '|' + '|'.join(['---'] * len(header)) + '|']
            md += ['| ' + ' | '.join(b) + ' |' for b in body] + ['']
            label = r'\label{tab:dokl:bv}' if p == 0 and s == 0 else ''
            tex += [r'\begin{table}[ht]', r'\centering', rf'\caption{{{title}}}{label}',
                    r'\begin{tabular}{l' + ' r' * (len(header) - 1) + '}', r'\toprule',
                    ' & '.join(header) + r' \\', r'\midrule']
            tex += [' & '.join(b) + r' \\' for b in body]
            tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    with open(f'{stem}.md', 'w') as f:
        f.write('\n'.join(md))
    with open(f'{stem}.tex', 'w') as f:
        f.write('\n'.join(tex))
    print(f'  saved {os.path.basename(stem)}.{{md,tex}}')


def counts(d, ks):
    """counts of the mse difference over the families, p*, N* and the kl indices ks.

    keys: cases; up / dn = dokls mse higher / lower, var / b2 = dokls variance /
    bias^2 larger, each with a clear count (_c); half and med = over the cases with a
    higher dokls mse, the variance part > half of the difference and the median
    variance share in %; cb / cv = over the cases with a clearly higher dokls mse, the
    bias^2 / variance part clear and positive.
    """
    db, dv = d['dbias2'][..., ks], d['dvar'][..., ks]
    dm = db + dv
    up = dm > 0
    cu = d['dmse_lo'][..., ks] > 0
    cb2, cvar = d['dbias2_lo'][..., ks] > 0, d['dvar_lo'][..., ks] > 0
    share = dv[up] / dm[up]
    return dict(cases=dm.size, up=up.sum(), up_c=cu.sum(), dn=(dm < 0).sum(),
                dn_c=(d['dmse_hi'][..., ks] < 0).sum(), var=(dv > 0).sum(),
                var_c=cvar.sum(), b2=(db > 0).sum(), b2_c=cb2.sum(),
                half=(share > 0.5).sum(),
                med=100 * np.median(share) if share.size else np.nan,
                cb=(cu & cb2).sum(), cv=(cu & cvar).sum())


def plain(s):
    """latex label -> markdown text."""
    for a, b in (('$^a$', ' (a)'), ('$^b$', ' (b)'), ('$^2$', '^2'), ('$>$', '>'),
                 ('$\\le 1$', '<= 1'), ('$\\ge 18$', '>= 18'), ('$3$, $9$', '3, 9'),
                 ('\\%', '%')):
        s = s.replace(a, b)
    return s


def summary(d, stem):
    """{stem}.md and {stem}.tex: counts() per kl range, one row per statistic.

    the tex is written in the normalized appendix style (normalize_tables would pad
    the integer counts, so this file is named without '_table').
    """
    bands = [('KL $\\le 1$', [0, 1]), ('KL $3$, $9$', [2, 3]), ('KL $\\ge 18$', [4, 5, 6]),
             ('All KL', list(range(len(d['kl']))))]
    spec = [('Cases', '{cases}'),
            ('MSE higher for DoKLs (clear)', '{up} ({up_c})'),
            ('MSE lower for DoKLs (clear)', '{dn} ({dn_c})'),
            ('Variance larger for DoKLs (clear)', '{var} ({var_c})'),
            ('Bias$^2$ larger for DoKLs (clear)', '{b2} ({b2_c})'),
            ('Variance part $>$ half of the MSE difference$^a$', '{half}'),
            ('Median variance share of the MSE difference$^a$', '{med:.0f}\\%'),
            ('Bias$^2$ part clear and positive$^b$', '{cb}'),
            ('Variance part clear and positive$^b$', '{cv}')]
    cs = [counts(d, ks) for _, ks in bands]
    body = [[label] + [form.format(**c) for c in cs] for label, form in spec]
    header = [''] + [name for name, _ in bands]
    cap = ('DoKLs ablation: DoKLs vs plug-in MSE over the six shared families, both test '
           'distributions and the three budgets $N_*$ (36 cases per KL level). In '
           'parentheses: the number of clear differences. $^a$Over the cases with a higher '
           'DoKLs MSE. $^b$Over the cases with a clearly higher DoKLs MSE')
    tex = ['% generated by ex/ablations/dokls/plot_bias_var.py from bias_var.h5',
           r'\begin{table}[H]\scriptsize', r'\centering',
           rf'\caption{{{cap}}}\label{{tab:dokl:bv:summary}}',
           r'\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{'
           r'\begin{tabular}{l' + ' r' * len(bands) + '}', r'\toprule',
           ' & '.join(rf'\textsc{{{h}}}' if h else '' for h in header) + r' \\',
           r'\midrule']
    tex += [' & '.join(r) + r' \\' for r in body]
    tex += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    md = ['| ' + ' | '.join(plain(h) for h in header) + ' |',
          '|' + '|'.join(['---'] * len(header)) + '|']
    md += ['| ' + ' | '.join(plain(c) for c in r) + ' |' for r in body]
    with open(f'{stem}.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open(f'{stem}.tex', 'w') as f:
        f.write('\n'.join(tex))
    print(f'  saved {os.path.basename(stem)}.{{md,tex}}')


def main():
    """bias_var.h5 -> two packed figures + the mse-difference table + its summary."""
    _, _, cfg = resolve('q0_N8192', 'two_leg')
    d = load(os.path.join(cfg['processed_results_dir'], 'bias_var.h5'))
    out = cfg['figures_dir']
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    for band, ks in BANDS.items():
        figure(d, ks, os.path.join(out, f'dokls_bv_kl_{band}'))
    table(d, os.path.join(out, 'dokls_bv_table'))
    summary(d, os.path.join(out, 'dokls_bv_summary'))


if __name__ == '__main__':
    main()
