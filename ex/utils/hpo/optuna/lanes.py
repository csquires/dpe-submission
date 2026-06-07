"""Lane registry for lane-aware HPO double-ended drain.

Defines compute profiles (LaneProfile) for distinct execution contexts:
- array:   cpu, many small slurm elements; <=96 (array_qos MaxJobsPU=100).
- cpu:     cpu qos, fat loky-fanout elements; <=9 (cpu_qos MaxJobsPU=10 minus the single redis-server job, which now also hosts the keepers).
- general: gpu, non-preemptible single-process; <=6 (normal-qos MaxTRESPU gpu=8).
- preempt: gpu, preemptible single-process; <=22 (preempt_qos MaxJobsPU=24).

Note: general and preempt require gpus>=1.

Provides registry accessors (get_lane) to retrieve resource config
(partition, qos, gpus, cpus_per_task, mem, batch_size, worker_walltime,
max_concurrent) to parameterize sbatch and trial dispatch.

Batch size semantics:
  - batch_size=None: derive at runtime as cpus_per_task // cores_per_trial(method).
  - batch_size=<int>: pin to that value, ignore per-method core allocation.

Cores-per-trial semantics:
  - cores_per_trial=None: use cores_registry.get_cores_for_method(method)
    (per-method latency-bound default).
  - cores_per_trial=<int>: pin to this value, ignore the registry. Use this
    when the lane has been profiled and the throughput-optimal shape differs
    from the per-method latency-bound default (e.g. array lane wants B=32
    cores=1, not B=8 cores=4).

QoS semantics:
  - qos="": omit --qos flag from sbatch (use partition default).
  - qos=<string>: pass --qos <string> to sbatch.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LaneProfile:
    """Compute profile for an HPO execution lane.

    Attributes:
        partition: sbatch partition name (e.g., 'array', 'preempt', 'general').
        qos: quality-of-service string (e.g., 'preempt_qos', 'cpu_qos').
             Empty string '' means omit --qos flag; use partition default.
        gpus: number of gpus per task (0 = cpu-only, 1+ = gpu).
        cpus_per_task: number of cpus allocated per sbatch task.
        mem: memory allocation string (e.g., '32G', '256G').
        batch_size: number of concurrent trials per worker.
                    None = derive at runtime as cpus_per_task // cores_per_trial(method).
                    int = pin to this value.
        cores_per_trial: BLAS threads per worker (overrides cores_registry).
                    None = look up per-method via cores_registry.
                    int = pin (used for throughput-bound lanes like 'array').
        worker_walltime: per-worker walltime limit, HH:MM:SS format.
        max_concurrent: per-lane TOTAL concurrent-job cap (the binding qos
                       MaxJobsPU/MaxTRESPU minus headroom). The keeper splits it
                       evenly across the studies still under target.
    """
    partition: str
    qos: str
    gpus: int
    cpus_per_task: int
    mem: str
    batch_size: int | None
    worker_walltime: str
    max_concurrent: int
    cores_per_trial: int | None = None


LANES: dict[str, LaneProfile] = {
    # array: profiled-optimal throughput shape (scratch/holdout_sweep_aggregate.py
    # + the zen 4/5 reps in scratch/holdout_profile_results_zen45/): cpus=32,
    # B=32, cores=1 -> 13.28 elem/min on epyc 7763, 1.43x cpus=16/B=8. consumers
    # (submit.py worker, submit_holdout.sh) read the lane to avoid duplicating
    # the shape.
    "array": LaneProfile(
        partition="array",
        qos="",
        gpus=0,
        cpus_per_task=32,
        mem="128G",
        batch_size=32,
        worker_walltime="18:00:00",
        max_concurrent=96,
        cores_per_trial=1,
    ),
    # holdout: same array partition + cores=1 shape as "array", but a smaller
    # 8-cpu/8-trial footprint so each slurm element backfills into scattered
    # idle cores. Only submit_holdout.sh reads this; the optuna worker stays
    # on "array".
    "holdout": LaneProfile(
        partition="array",
        qos="",
        gpus=0,
        cpus_per_task=8,
        mem="64G",
        batch_size=8,
        worker_walltime="18:00:00",
        max_concurrent=96,
        cores_per_trial=1,
    ),
    # cpu: fat loky-fanout, throughput-shaped to cores=1 like the array lane.
    # blas over-threading makes cores>1 ~10x worse total throughput (VFM profile:
    # c32/b32/cores1 = 425 elem/min vs c32/b8/cores4 = 39). cpus=48 fits the cpu
    # partition's largest single-node hole; mem scaled to the array lane's proven
    # ~4G/trial envelope at b48.
    "cpu": LaneProfile(
        partition="cpu",
        qos="cpu_qos",
        gpus=0,
        cpus_per_task=48,
        mem="128G",
        batch_size=None,
        worker_walltime="18:00:00",
        max_concurrent=9,
        cores_per_trial=1,
    ),
    "general": LaneProfile(
        partition="general",
        qos="",
        gpus=1,
        cpus_per_task=4,
        mem="32G",
        batch_size=1,
        worker_walltime="18:00:00",
        max_concurrent=6,
    ),
    "preempt": LaneProfile(
        partition="preempt",
        qos="preempt_qos",
        gpus=1,
        cpus_per_task=4,
        mem="64G",
        batch_size=32,
        worker_walltime="09:00:00",
        max_concurrent=22,
    ),
}


def get_lane(name: str) -> LaneProfile:
    """Retrieve a lane profile by name.

    Lanes are registered static profiles for distinct execution contexts.
    On KeyError, report known lane names to aid debugging.

    Args:
        name: lane profile name (e.g., 'array', 'cpu', 'general', 'preempt').

    Returns:
        LaneProfile: the named lane configuration.

    Raises:
        KeyError: if name not in LANES. message lists known lanes.
    """
    if name not in LANES:
        known = ", ".join(sorted(LANES.keys()))
        raise KeyError(f"lane '{name}' not found; known lanes: {known}")
    return LANES[name]
