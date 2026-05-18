"""Lane registry for lane-aware HPO double-ended drain.

Defines compute profiles (LaneProfile) for distinct execution contexts:
- array:   cpu, many small slurm elements; <=96 (array_qos MaxJobsPU=100).
- cpu:     cpu qos, fat loky-fanout elements; <=7 (cpu_qos MaxJobsPU=10 minus 3 keeper slots).
- general: gpu, non-preemptible single-process; <=6 (normal-qos MaxTRESPU gpu=8).
- preempt: gpu, preemptible single-process; <=22 (preempt_qos MaxJobsPU=24).

Note: general and preempt require gpus>=1.

Provides registry accessors (get_lane) to retrieve resource config
(partition, qos, gpus, cpus_per_task, mem, batch_size, worker_walltime,
max_concurrent) to parameterize sbatch and trial dispatch.

Batch size semantics:
  - batch_size=None: derive at runtime as cpus_per_task // cores_per_trial(method).
  - batch_size=<int>: pin to that value, ignore per-method core allocation.

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


LANES: dict[str, LaneProfile] = {
    "array": LaneProfile(
        partition="array",
        qos="",
        gpus=0,
        cpus_per_task=16,
        mem="64G",
        batch_size=None,
        worker_walltime="06:00:00",
        max_concurrent=96,
    ),
    "cpu": LaneProfile(
        partition="cpu",
        qos="cpu_qos",
        gpus=0,
        cpus_per_task=64,
        mem="32G",
        batch_size=None,
        worker_walltime="06:00:00",
        max_concurrent=7,
    ),
    "general": LaneProfile(
        partition="general",
        qos="",
        gpus=1,
        cpus_per_task=4,
        mem="32G",
        batch_size=1,
        worker_walltime="06:00:00",
        max_concurrent=6,
    ),
    "preempt": LaneProfile(
        partition="preempt",
        qos="preempt_qos",
        gpus=1,
        cpus_per_task=4,
        mem="32G",
        batch_size=1,
        worker_walltime="03:00:00",
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
