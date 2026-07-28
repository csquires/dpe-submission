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
        constraint: sbatch --constraint node feature (e.g. a gpu-type feature).
                    Empty string '' means omit --constraint.
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
    constraint: str = ""


LANES: dict[str, LaneProfile] = {
    # array: profiled-optimal throughput shape: cpus=32, B=32, cores=1 ->
    # 13.28 elem/min on epyc 7763, 1.43x cpus=16/B=8. consumers (submit.py
    # worker, submit_holdout.sh) read the lane to avoid duplicating the shape.
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
    # array_small: the "array" lane's cores=1 shape at the holdout lane's small
    # 8-cpu/8-trial footprint, for optuna workers when the array partition is
    # fragmented. cpus=32 is throughput-optimal per element but needs 32
    # contiguous idle cores; when the partition is saturated (e.g. ~600 idle
    # cores scattered over allocated nodes) those jobs sit PD "Priority"
    # indefinitely, while 8-cpu elements backfill and actually run. ~1.4x worse
    # per element, far better wall-clock when nothing else schedules.
    "array_small": LaneProfile(
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
        max_concurrent=24,
        # pin to newer high-vram gpu classes only. older/smaller cards (A6000,
        # A100_40GB, L40 non-S, H100/H200) get skipped -- keeps OOM risk down
        # for heavy score/flow methods and avoids preempt spins on nodes we
        # would rather not schedule on.
        constraint="L40S|RTX_PRO_6000|6000Ada",
    ),
    # array_gpu: gpu-on-array lane. gpus=1 on the array partition to use the
    # 8-gpu fanout allowed there, pinned to RTX_PRO_6000 (96G VRAM) nodes.
    # B=1: one trial per gpu. profiled 2026-07-05 -- B=8 (8 loky trials/gpu) ran
    # 2.2x SLOWER at steady state (1.8 vs 4 completes/min across 7 studies): the
    # dim-64 vfm/fmdre trials saturate the gpu so concurrency just time-slices,
    # and 56 workers contend on the shared redis journal (comms-bound).
    # dispatched as a job array (the array partition only accepts arrays) via
    # keeper.dispatch_array_gpu; the cpu "array" lane is left untouched.
    "array_gpu": LaneProfile(
        partition="array",
        qos="",
        gpus=1,
        cpus_per_task=4,
        mem="32G",
        batch_size=1,
        worker_walltime="18:00:00",
        max_concurrent=8,
        constraint="RTX_PRO_6000",
    ),
    # array_gpu_wide: like array_gpu but B=8 loky trials per gpu. for
    # LOW-DIM experiments (e.g. model_selection dim-3) where one trial barely
    # touches the gpu, so packing several per card should win even though each
    # loky worker is its own redis client (the journal-contention tradeoff is
    # the point of the experiment). routed via dispatch_array_gpu (partition
    # array + gpus>0), same as array_gpu.
    "array_gpu_wide": LaneProfile(
        partition="array",
        qos="",
        gpus=1,
        cpus_per_task=8,
        mem="32G",
        batch_size=8,
        worker_walltime="18:00:00",
        max_concurrent=8,
        constraint="RTX_PRO_6000",
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


def resolve_constraint(lane: LaneProfile) -> str:
    """Resolve constraint for a lane, respecting DPE_PREEMPT_CONSTRAINT override.

    If the lane partition is "preempt" and env var DPE_PREEMPT_CONSTRAINT is set,
    use that value (even "" to disable). Otherwise use lane.constraint.
    This allows per-run constraint tuning without modifying the lane registry.

    Args:
        lane: LaneProfile instance.

    Returns:
        Constraint string (may be empty "").
    """
    import os
    if lane.partition == "preempt" and "DPE_PREEMPT_CONSTRAINT" in os.environ:
        return os.environ["DPE_PREEMPT_CONSTRAINT"]
    return lane.constraint
