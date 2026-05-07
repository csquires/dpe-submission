# Gold-Standard Hyperparameter Winners

One yaml per experiment with the deployable HP set per method.

Generated 2026-05-07T02:01:46Z.

## Files

| experiment | non-tri | tri |
|---|---|---|
| dbpedia_cond_flow | 6 | 0 |
| dre_sample_complexity | 7 | 0 |
| eig_estimation | 8 | 0 |
| elbo_estimation | 8 | 8 |
| mnist_cond_flow | 8 | 8 |
| mnist_eldr_estimation | 6 | 0 |
| model_selection | 8 | 8 |
| pendulum_eldr_estimation | 7 | 0 |
| plugin_dre | 7 | 0 |
| smodice_eldr_estimation | 8 | 8 |

## Selection rules

- **non-triangular**: best HP set from the 200broad/r24/summary pool, evaluated on 200broad holdout grid (matches the natural 200broad eval).
- **triangular** (r24-scope experiments only — elbo, mnist_cond_flow, model_selection, smodice): best HP set from r24 / r24_upd1, evaluated on r24 holdout grid.
- triangular variants that use `integration_steps`: filtered to `integration_steps < 2000`.
- FMDRE / FMDRE_S2: additionally filtered to `integration_steps < 1750` per the upstream constraint.

## Schema per method entry

```yaml
methods:
  CTSM:
    hyperparams: {n_epochs: 894, lr: 0.00153, ...}
    score:
      mae: 0.569
      eval: 200broad_holdout       # or r24_holdout for triangular
      source: rank200/CTSM
      trial_id: 191
```

## MAE comparability

- non-tri rows are mutually comparable (all on 200broad_holdout).
- tri rows are mutually comparable (all on r24_holdout).
- Do NOT compare a tri MAE against a non-tri MAE directly — the eval grids differ.
