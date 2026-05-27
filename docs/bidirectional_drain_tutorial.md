# Bidirectional-Drain HPO Workflow: Setup Guide

## What This Workflow Does

A single watchdog process manages a shared queue file. GPU-side jobs pop trials from the **front** (slow methods that need GPU). CPU-side array jobs pop trials from the **back** (fast methods that run well on CPUs). Both sides use atomic flock operations on the same queue file, achieving ~2x throughput compared to GPU-only campaigns with no additional infrastructure.

**Key property**: both partitions feed from the same queue, sorted by method speed class, so neither side starves.

---

## Cluster Prerequisites

Before starting, verify your cluster has SLURM, the required partitions, and QOS limits. Note that these are anonymized sample names and may differ for your cluster. Read your cluster's documentation or contact its administrators. To investigate yourself, these commands may be helpful:

```bash
# Check GPU/preempt partition exists and has GPUs
sinfo -p preempt -N -o "%.6N %.10P %.5D %.5c %.10m"

# Check CPU/array partition exists
sinfo -p array -N -o "%.6N %.10P %.5D %.5c %.10m"

# Verify QOS limits (look for MaxJobsPU, MaxTRESPU)
sacctmgr show qos array_qos format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU%40

# Check SLURM array size limits
scontrol show config | grep -E "MaxArraySize|MaxJobCount"
```

**Required minimums**:
- preempt partition: ≥ 1 GPU node, MaxJobsPU ≥ 10
- array partition: ≥ 4 CPU nodes, MaxJobsPU ≥ 30
- SLURM MaxArraySize ≥ 1000
- shared NFS filesystem (for queue file and results, e.g., `/data`, `/home`)

When you find your cluster's metadata (e.g., `gpu` instead of `preempt`, `cpu` instead of `array`), note those names now. You may need to adapt them in the code files below. (Along with the resource limits). 

---

## One-Time Configuration

### 1. Method Speed Classification

Edit `$DPE_WORKDIR/experiments/utils/walltime_caps.py`.

The `SPEED_CLASS_MAP` dict classifies all methods:
- **SLOW** (0): GPU-only, long runtimes (flow integration). Front of queue.
- **MEDIUM** (1): CPU-eligible, 5–10 min per trial. Mid queue.
- **FAST** (2): CPU-eligible, 3–5 min per trial. Back of queue.

Check or update method classifications. Example:
```python
SPEED_CLASS_MAP: dict[str, SpeedClass] = {
    "VFM":                       SpeedClass.SLOW,   # flow-based, gpu-only
    "TSM":                       SpeedClass.MEDIUM,  # cpu-eligible
    "MDRE_15":                   SpeedClass.FAST,    # tabular, cheap
    # ... add your custom methods here
}
```

### 2. Walltime Caps

In the same file, configure per-method walltime limits:

**WALLTIME_CAPS_PREEMPT**: per-trial cap on GPU (preempt partition). Used by watchdog.
```python
WALLTIME_CAPS_PREEMPT = {
    "TSM":   "0:05:00",    # observed 4-cell median 15s; cap 5x
    "VFM":   "1:15:00",    # flow integration; cap at p99 rate
    # ...
}
```

**WALLTIME_CAPS_CPU**: per-trial cap on CPU (array partition). Used by cpu_dispatcher.
- **Only include CPU-eligible methods** (MEDIUM and FAST from SPEED_CLASS_MAP).
- **Do NOT include SLOW methods** (they will crash on CPU).
- Typically 2–3x preempt cap (non-preempt jobs need slack).

```python
WALLTIME_CAPS_CPU = {
    "TSM":                  "0:10:00",
    "CTSM":                 "0:10:00",
    "MDRE_15":              "0:05:00",
    # ... only cpu-eligible methods
}
```

### 3. Partition Names

Edit `$DPE_WORKDIR/experiments/utils/submit_watchdog.sh`:
```bash
# Line 46: replace 'cpu' with your CPU partition name if different
sbatch --parsable \
    --partition=cpu \  # <-- change if your cluster uses 'array', 'compute', etc.
    ...
```

Edit `$DPE_WORKDIR/experiments/utils/hpo/cpu_dispatcher.py`:
```python
# Line 72: replace 'array' with your CPU partition name if different
cmd = [
    "sbatch",
    ...
    "--partition=array",  # <-- change if your cluster uses 'cpu', 'compute', etc.
    ...
]
```

### 4. Project Paths

Project paths are resolved from environment variables; you do not need to edit
any source files. See the root `README.md` for the full list. The relevant
variable for this workflow is `DPE_WORKDIR`, which points at the repo root.

### 5. Environment Setup

Define or export these env vars **before launching**:

```bash
# Repo root (defaults to repo's location resolved from package __file__)
export DPE_WORKDIR=/path/to/dpe-submission

# Shared directory for queue file, results, and manifests
# (use NFS on cluster, $HOME/dpe-data locally)
export DPE_DATA_ROOT=/path/to/dpe/data

# Optional: node-local scratch for checkpoints (cross-node recovery via NFS queue)
export DPE_CKPT_ROOT=/path/to/dpe/ckpt

# Conda environment (default: 'fac')
conda activate "${DPE_CONDA_ENV:-fac}"
```

Verify connectivity:
```bash
ls -ld $DPE_DATA_ROOT
mkdir -p $DPE_DATA_ROOT
touch $DPE_DATA_ROOT/test_write.txt && rm $DPE_DATA_ROOT/test_write.txt
echo "NFS mounted and writable."
```

---

## Experiment Adapters

Each experiment has an adapter under `experiments/utils/hpo/adapters/<name>.py` that subclasses `ExperimentAdapter` (defined in `base.py`).

**Required interface**:

| Method | Purpose |
|--------|---------|
| `name()` | Experiment identifier (e.g., "mnist_cond_flow") |
| `cell_pool()` | List of tuples, e.g., `[(0,0), ..., (3,39)]` |
| `load_cell_data(cell, device)` | Load cell from H5; return dict with keys `pstar`, `p0`, `p1`, `true_ldrs` |
| `metric_key()` | Result JSON key for the loss metric, e.g., "per_pair_mae" |
| `latent_dim()` | Input dimension for estimators |
| `num_waypoints()` | Waypoints for triangular methods, or None |
| `device()` | "cpu" or "cuda" |
| `default_training_M()` | Default training cell count (usually 32) |
| `default_holdout_M()` | Default holdout cell count (usually 32) |
| `is_ready()` | True iff data is available and ready |
| `supports_tabular()` | True iff experiment supports tabular-only methods (rare) |

**Example: minimal adapter**

```python
# experiments/utils/hpo/adapters/my_experiment.py
from pathlib import Path
from experiments.utils.hpo.adapters.base import ExperimentAdapter
import torch

class MyExperimentAdapter(ExperimentAdapter):
    """My experiment: 3-cell grid × 4 seeds per cell = 12 total cells. 1D tuples."""
    
    def name(self) -> str:
        return "my_experiment"
    
    def cell_pool(self) -> list[tuple[int, ...]]:
        # 3 × 4 grid
        return [(i, j) for i in range(3) for j in range(4)]
    
    def load_cell_data(self, cell, device):
        path = self.data_dir() / f"cell_{cell[0]}_{cell[1]}.h5"
        # Load from HDF5; return dict with required keys
        return {
            "pstar": torch.randn(10, device=device),
            "p0": torch.randn(10, device=device),
            "p1": torch.randn(10, device=device),
            "true_ldrs": torch.randn(10, device=device),
        }
    
    def metric_key(self) -> str:
        return "my_metric_mae"
    
    def latent_dim(self) -> int:
        return 10
    
    def num_waypoints(self):
        return None  # or 5, 10, etc.
    
    def device(self) -> str:
        return "cuda"
    
    def data_dir(self) -> Path:
        return Path(f"{os.environ.get('DPE_DATA_ROOT', '/data')}/my_experiment")
```

**Register the adapter** in `experiments/utils/hpo/adapters/__init__.py`:
```python
from experiments.utils.hpo.adapters.my_experiment import MyExperimentAdapter

_ADAPTERS = {
    "my_experiment": MyExperimentAdapter,
    # ... other adapters
}
```

---

## Launching a Campaign

### Quick Start

```bash
cd "$DPE_WORKDIR"
export DPE_DATA_ROOT=/path/to/dpe/data
mkdir -p $DPE_DATA_ROOT

# Test run: 5 CPU elements (small pilot)
python -m experiments.utils.hpo.launcher \
    --experiments mnist_cond_flow,smodice_eldr_estimation \
    --methods all \
    --my-cap 24 --total-cap 60 \
    --cpu-array-max 5 \
    --cpu-concurrency 4 \
    --cpu-cpus-per-task 4 \
    --cpu-n-per-element 8 \
    --no-cpu-drain
```

Wait 5–10 minutes for GPU jobs to complete. Monitor:
```bash
squeue -u $USER
tail -f $DPE_DATA_ROOT/*/watchdog/*/watchdog.log
```

### Full Production Campaign

Once pilot succeeds:

```bash
python -m experiments.utils.hpo.launcher \
    --experiments mnist_cond_flow,smodice_eldr_estimation \
    --methods all \
    --my-cap 24 --total-cap 60 \
    --cpu-walltime auto \
    --cpu-concurrency 64 \
    --cpu-cpus-per-task 4 \
    --cpu-n-per-element 8
```

**Flag reference**:

| Flag | Default | Purpose |
|------|---------|---------|
| `--experiments` | "all" | Comma-sep list: `mnist_cond_flow,pendulum` or `all` |
| `--methods` | "all" | Comma-sep list: `TSM,BDRE,MDRE_15` or `all` |
| `--my-cap` | 80 | Per-user preempt job limit |
| `--total-cap` | 200 | Total preempt job limit (shared) |
| `--wave-size` | 3 | Workflows per submission wave (pacing, not concurrency) |
| `--budget` | 250 | Trials per (method, exp) pair |
| `--cpu-walltime` | "1:30:00" | Per-element walltime; `auto` to compute from spec |
| `--cpu-concurrency` | 100 | Max concurrent array elements |
| `--cpu-n-per-element` | 8 | Trials per array element |
| `--cpu-cpus-per-task` | 2 | CPUs per element |
| `--cpu-mem` | "8G" | Memory per element |
| `--cpu-array-max` | 200 | Array size cap |
| `--no-cpu-drain` | false | Skip CPU array (GPU-only campaign) |
| `--dry-run` | false | Print sbatch commands without submitting |

---

## Artifacts and Results

### Directory Structure

```
$DPE_DATA_ROOT/
├── multi_experiment_watchdog_queue.txt  # shared queue file
├── <exp_name>/
│   ├── recalibrated_specs/
│   │   └── <exp_name>.yaml              # optional: search-space narrowing
│   ├── <method_name>/
│   │   └── broad/
│   │       ├── trial_0.json
│   │       ├── trial_1.json
│   │       └── ...                      # per-trial results (atomic write)
│   └── winners.<exp_name>.yaml          # persisted winners across trials
└── watchdog/<run_id>/
    ├── launcher_manifest.json           # campaign metadata + job IDs
    ├── watchdog.log                     # watchdog process log
    ├── watchdog.out                     # watchdog stdout
    ├── watchdog.err                     # watchdog stderr
    └── cpu_array_logs/
        ├── <exp_tag>_0000.out
        ├── <exp_tag>_0001.out
        └── ...                          # per-element stdout/stderr
```

### Key Files

**trial_<id>.json**: one trial's result. Example:
```json
{
  "cell": [0, 0],
  "method": "TSM",
  "trial_id": 42,
  "per_pair_mae": 0.0234,
  "runtime_sec": 12.5,
  "timestamp": "2025-05-01T14:23:45Z"
}
```

**launcher_manifest.json**: campaign metadata.
```json
{
  "run_id": "20250501_142300",
  "watchdog_jid": 12345678,
  "workflow_jids": [12345679, 12345680, ...],
  "cpu_array_dispatched": true,
  "cpu_array_jid": 12345700,
  "cpu_array_size": 64,
  "cpu_array_experiments": ["mnist_cond_flow", "smodice_eldr_estimation"],
  "timestamp": "2025-05-01T14:23:00Z"
}
```

---

## Monitoring

### Live Status

```bash
# All jobs for current user
squeue -u $USER -o "%.10i %.9P %.30j %.2t %.10M"

# Watchdog job only
squeue -j <watchdog_jid>

# Queue depth (how many trials remain)
wc -l $DPE_DATA_ROOT/multi_experiment_watchdog_queue.txt

# Total trials landed (atomic .json files)
find $DPE_DATA_ROOT -name "trial_*.json" | wc -l

# Trials per method per experiment
find $DPE_DATA_ROOT -name "trial_*.json" | while read f; do
  dir=$(dirname "$f")
  echo "$dir"
done | sort | uniq -c
```

### Watchdog Logs

```bash
# Real-time watchdog activity
tail -f $DPE_DATA_ROOT/<exp>/watchdog/<run_id>/watchdog.log

# CPU array element log (e.g., element 5)
tail -f $DPE_DATA_ROOT/<exp>/watchdog/<run_id>/cpu_array_logs/<exp_tag>_00005.out

# Grep for errors
grep -i "error\|traceback\|exception" $DPE_DATA_ROOT/*/watchdog/*/watchdog.log
```

### Check Completeness

```bash
# Expected trials per (method, exp) pair
n_methods=$(echo "TSM,BDRE,MDRE_15" | tr ',' '\n' | wc -l)
n_exps=$(echo "mnist_cond_flow,smodice_eldr_estimation" | tr ',' '\n' | wc -l)
budget=250
expected=$((n_methods * n_exps * budget))
echo "Expected trials: $expected"

# Actual
actual=$(find $DPE_DATA_ROOT -name "trial_*.json" | wc -l)
echo "Actual trials: $actual"
```

---

## Troubleshooting

### GPU Jobs (Watchdog) Stall

**Symptom**: Queue has 100+ lines, watchdog log shows no new job submissions.

**Check**:
- `squeue | grep preempt` — are jobs running?
- `sacctmgr show qos preempt_qos` — is the QOS capped? (increase --my-cap or --total-cap)
- Watchdog is preempted: check `squeue -j <watchdog_jid>` and logs; automatic requeue restarts it.

**Action**: increase `--my-cap` or wait for other users' jobs to finish.

### CPU Array Elements TIMEOUT

**Symptom**: `tail -f cpu_array_logs/<tag>_00010.out` shows partial results, then job killed at walltime.

**Cause**: walltime cap too tight for method(s) in the filter.

**Action**: 
- Increase `--cpu-walltime` (e.g., from "1:30:00" to "2:00:00"), or
- Check `WALLTIME_CAPS_CPU` for the method; bump the cap, or
- Reduce `--cpu-n-per-element` (fewer trials per element = shorter runtime).

### CPU Array Submits 0 Elements

**Symptom**: Launcher output shows "cpu array dispatched: false" with "no_eligible_methods".

**Cause**: All methods queued are SLOW (GPU-only); none in MEDIUM/FAST classes.

**Check**:
- `SPEED_CLASS_MAP` in `walltime_caps.py` — verify your methods are classified as MEDIUM or FAST.
- Check queue file: is it only VFM, FMDRE, etc.? If so, this is expected; CPU cannot run them.

**Action**: Run a mixed campaign: `--methods TSM,VFM` (include at least one CPU-eligible method).

### Results Not Landing on NFS

**Symptom**: launcher_manifest.json shows trials landed (trial_*.json files exist), but you see a disk error in logs.

**Cause**: 
- NFS unmounted mid-campaign.
- DPE_DATA_ROOT points to /tmp or /scratch (node-local, not NFS).
- File permissions on NFS (group write not allowed).

**Check**:
```bash
mount | grep -E "/data|/home"
ls -ld $DPE_DATA_ROOT
touch $DPE_DATA_ROOT/test && rm $DPE_DATA_ROOT/test  # must succeed
```

**Action**: Set `DPE_DATA_ROOT` to an NFS mount and re-launch.

### Watchdog Crashes and Restarts

**Symptom**: watchdog.log shows repeated "Traceback" → restart after ~60 sec.

**Cause**: depends on error; common ones:
- Bad JSON in queue file (malformed trial spec).
- Missing adapter or incomplete `load_cell_data()` implementation.
- GPU OOM (too many concurrent trials).

**Action**:
- Read watchdog.log fully; find the "Traceback" line.
- Fix the adapter or config and cancel the watchdog: `scancel <watchdog_jid>`.
- Re-launch. The watchdog auto-rescans results and skips completed trials.

### Wildly Mismatched Drain Rates

**Symptom**: GPU (watchdog) submits ~10 jobs/min, CPU (array) completes ~1 job/min.

**Cause**: CPU elements are slow (many trials, large models) or contending on shared nodes.

**Expected**: Under baseline (64 CPU elements, 8 trials/element, 4 cpus/element):
- ~2 nodes of 64 CPUs total → ~4 elements per node
- Each element runs 8 trials (~60 sec) sequentially
- Total time per node: 8 trials × 60 sec = ~480 sec
- Throughput: 64 elements / 480 sec ≈ 0.13 elements/sec ≈ 1 job/sec ≈ 60 jobs/min @ 8 trials/job

If you see ~10 jobs/min CPU side, you are in the nominal regime. No action needed.

If CPU drains much slower (e.g., 1 job/min), consider:
- `--cpu-n-per-element 4` (halve trials per element → faster throughput)
- `--cpu-cpus-per-task 8` (more resources per element, fewer elements sharing a node)
- `--ntasks-per-node 2` (restrict to 2 elements per node instead of default many)

---

## Validation Gates Before Production

### 5-Element Pilot

Before a multi-thousand-trial campaign, run a tiny pilot to validate schema parity:

```bash
python -m experiments.utils.hpo.launcher \
    --experiments mnist_cond_flow \
    --methods TSM,MDRE_15 \
    --my-cap 4 --total-cap 10 \
    --budget 2 \
    --cpu-array-max 5 \
    --cpu-concurrency 4 \
    --no-cpu-drain
```

Wait 2–5 minutes. Check:
```bash
find $DPE_DATA_ROOT/mnist_cond_flow -name "trial_*.json" | head -1 | xargs cat | python -m json.tool
# Should have keys: cell, method, trial_id, <metric_key>, runtime_sec, timestamp
```

If the JSON structure matches your downstream analysis, proceed to full campaign.

### n_jobs Equivalence (Optional)

If you vary `--cpu-n-per-element` (e.g., 8 vs. 4), CPU results should have similar quality. This is usually automatic if elements are i.i.d. trials; no separate validation needed unless you suspect element-level contention.

---

## Common Mistakes

### Mistake 1: Using node-local storage e.g. `/tmp` for the queue file

```bash
# WRONG: queue file lost if node reboots
export DPE_DATA_ROOT=/tmp/dpe-submission
python -m experiments.utils.hpo.launcher ...
```

**Fix**: Use NFS-mounted path. For example, on a cluster where NFS is mounted
at `/data/user_data`:
```bash
export DPE_DATA_ROOT=/data/user_data/$USER/dpe-submission
```

### Mistake 2: Hardcoded Usernames

```bash
# WRONG: breaks for other users
LOGDIR="/data/alice/dpe-submission"
```

**Fix**: Use `$USER` env var.
```bash
LOGDIR="$DPE_DATA_ROOT"   # already user-scoped via DPE_DATA_ROOT
```

### Mistake 3: GPU-Only Methods in CPU Filter

```python
# WRONG: VFM (SLOW) queued for CPU array
SPEED_CLASS_MAP["VFM"] = SpeedClass.SLOW  # correct
WALLTIME_CAPS_CPU = {
    "VFM": "1:15:00",  # WRONG; VFM will crash on CPU
    "TSM": "0:10:00",
}
```

**Fix**: Exclude SLOW methods from WALLTIME_CAPS_CPU.
```python
WALLTIME_CAPS_CPU = {
    "TSM": "0:10:00",
    "BDRE": "0:10:00",
    # No VFM, FMDRE, etc.
}
```

### Mistake 4: Using Relative Paths in Config

```python
# WRONG: breaks if cwd changes
WORKDIR = "."
cpu_dispatcher.py uses relative path to queue file
```

**Fix**: Use absolute paths.
```python
WORKDIR = "/home/user/dpe-submission"
queue_file = Path(os.environ["DPE_DATA_ROOT"]) / "multi_experiment_watchdog_queue.txt"
```

---

## Further Reading

- **Watchdog internals**: `experiments/utils/watchdog.py`
- **CPU dispatcher**: `experiments/utils/hpo/cpu_dispatcher.py` and `cpu_array_element.py`
- **Adapter base class**: `experiments/utils/hpo/adapters/base.py`
- **Method specs**: `experiments/utils/hpo/method_specs.py`
- **Walltime calibration**: `experiments/utils/walltime_caps.py` (docstrings for per-partition tuning)
