# cpu_runner — interactive CPU-only HPO

A standalone multiprocessed HPO runner for interactive Babel jobs. Replaces
the SLURM broad/refined stages when you want to iterate on CPU without the
watchdog+preempt machinery.

## Where it fits in the HPO flow

```
Full SLURM flow (launcher.py --stage all):
  recalibrate → broad (200 GPU trials) → refined (49 GPU trials) → holdout → persist
                ↑____________________________________________↑
                      cpu_runner covers these two stages
```

`cpu_runner` writes the same `trial_N.json` format as the SLURM path, so you
can feed its output into `workflow.py --stage holdout/persist` if needed.

## How it works

- **Preloads** all eval cell data once in the parent process before forking.
- **Forks** `n_jobs` worker processes (Linux copy-on-write — no re-loading per trial).
- Workers evaluate trials in parallel and print results as they complete.
- Writes `<output-dir>/<stage>/trial_N.json` atomically per trial.
- Writes `<output-dir>/summary.json` with a sorted leaderboard at the end.

CPU budget: approximately `n_jobs × inner_threads` cores.

## Works with any registered experiment

The runner is adapter-driven. Any experiment with an entry in
`adapters/__init__.py` works:

```
dre_sample_complexity   eig_estimation    mnist_estimation
elbo_estimation         pendulum_eldr_estimation   ...
```

## Quick smoke test (5 epochs, tiny model, 2 workers)

Request an interactive job first:

```bash
srun --partition=preempt --cpus-per-task=6 --mem=12G --time=0:30:00 --pty bash
```

Then in the session:

```bash
conda activate fac
cd /home/aviamala/dpe-submission

python -m experiments.utils.hpo.cpu_runner \
    --experiment dre_sample_complexity \
    --method MDRE \
    --n-trials 6 \
    --n-cells 5 \
    --n-jobs 2 \
    --inner-threads 2 \
    --output-dir /tmp/hpo_smoke \
    --override-hyperparams '{"num_epochs": 5, "latent_dim": 16}'
```

Expected output:

```
[cpu_runner] experiment='dre_sample_complexity'  method='MDRE_15'
[cpu_runner] n_trials=6  n_cells=5  n_jobs=2  inner_threads=2
[cpu_runner] estimated cpu usage: ~4 cores
[cpu_runner] preloading 5 cells...
[cpu_runner] preloaded in 0.8s
[cpu_runner] hyperparameter overrides: {'num_epochs': 5, 'latent_dim': 16}
[cpu_runner] dispatching 6 trials, 2 at a time...
[trial    1] MDRE_15: 5/5 cells, score=0.3241, 4.1s -> trial_1.json
[trial    0] MDRE_15: 5/5 cells, score=0.2891, 4.3s -> trial_0.json   ← out of order = parallel
[cpu_runner] 2/6 done  elapsed=5s  eta=10s  best_so_far=0.2891
...

=== top 6/6 results  (.../broad) ===
  rank  trial     score  mean_mae  hyperparams
     1      3    0.2341    0.2341  {'latent_dim': 16, 'learning_rate': ..., 'num_epochs': 5}
     2      0    0.2891    0.2891  ...
```

### Verifying it is actually running in parallel

1. **Trials finish out of order** — if trial 3 prints before trial 0, it's parallel.
2. **Elapsed time** — 6 trials with 2 workers should take ~3× a single trial's time,
   not 6×.
3. **From a second terminal while running**:
   ```bash
   # see 1 parent + 2 worker python processes
   ps aux | grep cpu_runner

   # watch finished trial count tick up in pairs
   watch -n2 "ls /tmp/hpo_smoke/broad/*.json 2>/dev/null | wc -l"
   ```

## Monitoring a longer run (second terminal)

```bash
# count finished trials
watch -n5 "ls <output-dir>/broad/*.json 2>/dev/null | wc -l"

# live leaderboard (re-run any time)
python -m experiments.utils.hpo.cpu_runner \
    --show-results --output-dir <output-dir>/broad

# top 5 only
python -m experiments.utils.hpo.cpu_runner \
    --show-results --output-dir <output-dir>/broad --top 5
```

## Full HPO run (no overrides)

```bash
python -m experiments.utils.hpo.cpu_runner \
    --experiment dre_sample_complexity \
    --method MDRE \
    --n-trials 50 \
    --n-cells 8 \
    --n-jobs 4 \
    --inner-threads 2 \
    --output-dir $DPE_DATA_ROOT/cpu_hpo/dre_sample_complexity/MDRE_15
```

### Resume after interruption

```bash
python -m experiments.utils.hpo.cpu_runner \
    ... same args ... \
    --resume      # skips trials whose trial_N.json already exists
```

## CLI reference

| flag | default | description |
|---|---|---|
| `--experiment` | required | adapter name (e.g. `dre_sample_complexity`) |
| `--method` | required | method name or alias (e.g. `MDRE`, `BDRE`, `TSM`) |
| `--n-trials` | 50 | total HPO trials |
| `--n-cells` | 8 | eval cells per trial |
| `--n-jobs` | 4 | concurrent workers (trials at once) |
| `--inner-threads` | 2 | BLAS threads per worker |
| `--seed` | 1729 | cell sampling seed |
| `--stage` | broad | output label (`broad` or `refined`) |
| `--output-dir` | required | results root; files go to `<dir>/<stage>/trial_N.json` |
| `--override-hyperparams` | — | JSON dict to pin hyperparams, e.g. `'{"num_epochs": 5}'` |
| `--resume` | — | skip already-finished trials |
| `--show-results` | — | print leaderboard from existing results and exit |
| `--top` | 10 | rows shown by `--show-results` |
