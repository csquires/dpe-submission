"""bias-variance split of the eldr error: two-leg vs plug-in at N = 8192.

procedure:
    two-leg raw (dokls) + dokls gt        --> e[row, p*, N*, kl, inst]
    plug-in raw (model_selection) + ms gt -/   (row = route x method)
    e per kl stratum (10 instances)       --> bias, var, mse, rmse + 95% boot ci,
                                              bias^2 share, n finite
    two-leg vs plug-in, per family        --> mse difference = bias^2 part + variance
                                              part, each with a paired 95% boot ci
    --> processed_results/bias_var.h5

definitions (finite instances only; ddof 0, so bias^2 + var = mse exactly):
    e_i = mean_j est(x_j) - eldr_i,  bias = mean_i e_i,
    var = mean_i (e_i - bias)^2,  mse = bias^2 + var,  share = bias^2 / mse,
    dmse = mse_two_leg - mse_plugin = dbias2 + dvar  (dbias2 = difference of bias^2,
    dvar = difference of var). a difference is clear when its ci excludes 0.

the dokls and model_selection datasets hold the same 70 (mu, Sigma) instances
(checked on load), so the routes are paired by instance. each instance has one
fit, so var is the spread over instances, not the sampling variance of one fit.
two-leg: mean over the p* samples that its legs were fit on. plug-in:
model_selection tr8192_te{N*}, mean over held-out p* samples (test set 2 = q0,
3 = q1), vs the analytic eldr.

usage: python -m ex.ablations.dokls.bias_var
"""
import h5py
import numpy as np

from ex.ablations.dokls import variants
from ex.ablations.dokls.plot_vs_N import MS_COMMON, MS_TESTIDX
from ex.synth.model_selection import variants as ms_variants

LOSSES = {'BDRE': ('BDRE_NWJ', 'BDRE_DV'), 'MultiHeadTDRE': ('MHT_NWJ', 'MHT_DV')}
NSTARS = [2048, 4096, 8192]
N_FIT = 8192
N_KL, N_INST = 7, 10
N_BOOT = 2000
PARAMS = ('mu0_arr', 'mu1_arr', 'Sigma0_arr', 'Sigma1_arr')


def rows():
    """[(route, method, family)] in table order: plug-in, two-leg, nwj, dv per family."""
    out = []
    for fam in MS_COMMON:
        out += [('plugin', fam, fam), ('two_leg', fam, fam)]
        out += [('two_leg', m, fam) for m in LOSSES.get(fam, ())]
    return out


def tag(p, ns):
    """dokls variant tag for p* index p and p* budget ns at N = 8192."""
    return f'q{p}_N{N_FIT}' + ('' if ns == N_FIT else f'_ns{ns}')


def two_leg(cfg, truth):
    """{(method, p, ns): e (70,)} for every two-leg row, vs the dokls analytic eldr."""
    methods = [m for r, m, _ in rows() if r == 'two_leg']
    out = {}
    for p in (0, 1):
        for ns in NSTARS:
            name = variants.experiment_name(tag(p, ns), 'two_leg')
            with h5py.File(f"{cfg['raw_results_dir']}/{name}_results.h5", 'r') as f:
                for m in methods:
                    est = f[f'est_ldrs_arr_{m}'][:]
                    out[(m, p, ns)] = est.mean(axis=1, dtype=np.float64) - truth[:, p]
    return out


def plugin(params):
    """{(method, p, ns): e (70,)} for the plug-in rows; checks the instance pairing.

    raises ValueError if a model_selection dataset holds other (mu, Sigma) than dokls.
    """
    out = {}
    for ns in NSTARS:
        _, cfg = ms_variants.resolve(f'tr{N_FIT}_te{ns}')
        with h5py.File(f"{cfg['data_dir']}/dataset_newpstar.h5", 'r') as f:
            if not all(np.array_equal(f[k][:], params[k]) for k in PARAMS):
                raise ValueError(f'tr{N_FIT}_te{ns}: instances differ from dokls')
            truth = f['true_eldr_analytic_arr'][:].astype(np.float64)
        with h5py.File(f"{cfg['raw_results_dir']}/new_pstar.h5", 'r') as f:
            for m in MS_COMMON:
                for p, t in MS_TESTIDX.items():
                    est = f[f'est_ldrs_arr_{m}'][:, t, :]
                    out[(m, p, ns)] = est.mean(axis=1, dtype=np.float64) - truth[:, t]
    return out


def split(e):
    """bias, var, mse, share, n over the last axis, from finite entries only.

    strata with no finite entry get nan (n = 0).
    """
    ok = np.isfinite(e)
    n = ok.sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        bias = np.where(ok, e, 0.0).sum(-1) / n
        var = np.where(ok, (e - bias[..., None]) ** 2, 0.0).sum(-1) / n
        mse = bias ** 2 + var
        share = bias ** 2 / mse
    return dict(bias=bias, var=var, mse=mse, share=share, n=n)


def boot_ci(e, rng, n_boot=N_BOOT):
    """(lo, hi): 95% percentile ci of rmse per stratum over the finite instances."""
    lo = np.full(e.shape[:-1], np.nan)
    hi = np.full(e.shape[:-1], np.nan)
    for idx in np.ndindex(*e.shape[:-1]):
        x = e[idx][np.isfinite(e[idx])]
        if x.size < 2:
            continue
        r = np.sqrt((x[rng.integers(0, x.size, (n_boot, x.size))] ** 2).mean(1))
        lo[idx], hi[idx] = np.percentile(r, [2.5, 97.5])
    return lo, hi


def diff_ci(a, b, rng, n_boot=N_BOOT):
    """paired 95% boot ci of the mse difference a - b and of its two parts.

    a, b: (..., n) errors of two routes on the same instances. one resample of the
    instances that are finite in both serves both routes. returns
    {'dmse' | 'dbias2' | 'dvar': (lo, hi)}; strata with fewer than 3 such
    instances get nan.
    """
    out = {k: (np.full(a.shape[:-1], np.nan), np.full(a.shape[:-1], np.nan))
           for k in ('dmse', 'dbias2', 'dvar')}
    for idx in np.ndindex(*a.shape[:-1]):
        ok = np.isfinite(a[idx]) & np.isfinite(b[idx])
        x, y = a[idx][ok], b[idx][ok]
        if x.size < 3:
            continue
        k = rng.integers(0, x.size, (n_boot, x.size))
        db = x[k].mean(1) ** 2 - y[k].mean(1) ** 2
        dv = x[k].var(1) - y[k].var(1)
        for name, z in (('dmse', db + dv), ('dbias2', db), ('dvar', dv)):
            out[name][0][idx], out[name][1][idx] = np.percentile(z, [2.5, 97.5])
    return out


def main():
    """load both routes, split per stratum, write processed_results/bias_var.h5."""
    _, _, cfg = variants.resolve(tag(0, N_FIT), 'two_leg')
    with h5py.File(f"{cfg['data_dir']}/dataset.h5", 'r') as f:
        truth = f['true_eldr_arr'][:].astype(np.float64)
        params = {k: f[k][:] for k in PARAMS}
    src = {'two_leg': two_leg(cfg, truth), 'plugin': plugin(params)}
    rs = rows()
    e = np.full((len(rs), 2, len(NSTARS), N_KL, N_INST), np.nan)
    for r, (route, m, _) in enumerate(rs):
        for p in (0, 1):
            for s, ns in enumerate(NSTARS):
                e[r, p, s] = src[route][(m, p, ns)].reshape(N_KL, N_INST)
    st = split(e)
    rng = np.random.default_rng(cfg.get('seed', 1729))
    lo, hi = boot_ci(e, rng)
    tl = [rs.index(('two_leg', f, f)) for f in MS_COMMON]
    pi = [rs.index(('plugin', f, f)) for f in MS_COMMON]
    diff = {'dbias2': st['bias'][tl] ** 2 - st['bias'][pi] ** 2,
            'dvar': st['var'][tl] - st['var'][pi]}
    diff['dmse'] = diff['dbias2'] + diff['dvar']
    ci = diff_ci(e[tl], e[pi], rng)
    out = f"{cfg['processed_results_dir']}/bias_var.h5"
    with h5py.File(out, 'w') as f:
        f['err'] = e
        for k in ('bias', 'var', 'mse', 'share', 'n'):
            f[k] = st[k]
        f['rmse'] = np.sqrt(st['mse'])
        f['rmse_lo'], f['rmse_hi'] = lo, hi
        for k, v in diff.items():
            f[k] = v
            f[f'{k}_lo'], f[f'{k}_hi'] = ci[k]
        f['rows'] = np.array([f'{r}/{m}' for r, m, _ in rs], dtype='S')
        f['row_family'] = np.array([fam for *_, fam in rs], dtype='S')
        f['families'] = np.array(MS_COMMON, dtype='S')
        f.attrs['axes'] = ('row, pstar, nstar, kl[, instance]; '
                           'd*: family, pstar, nstar, kl (two-leg minus plug-in)')
        f.attrs['pstar'] = [0, 1]
        f.attrs['nstar'] = NSTARS
        f.attrs['kl'] = cfg['kl_distances']
        f.attrs['n_fit'] = N_FIT
        f.attrs['n_boot'] = N_BOOT
        f.attrs['seed'] = cfg.get('seed', 1729)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
