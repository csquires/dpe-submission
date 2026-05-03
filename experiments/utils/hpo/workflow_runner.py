"""watchdog-managed workflow state machine.

extracts the per-(method, experiment) state machine from workflow.py main()
into a tickable form. one tick advances at most one stage; barriers checked
via output-dir glob counts. eliminates the need for separate workflow
controller cpu_qos jobs (one per method).

stages (sequential):
  init -> gen_broad -> wait_broad
       -> gen_refined -> wait_refined  (or skip on flat scores)
       -> gen_holdout -> wait_holdout
       -> gen_persist -> done

interaction model:
  - watchdog ticks each pair once per cycle
  - GEN_* stages call workflow.{recalibrate,broad,refined,holdout,persist}
    which write configs + sbatch lines to the shared queue file
  - WAIT_* stages just check len(<output_dir>/<stage>/trial_*.json) >= budget
  - watchdog dispatches the queued lines via existing pop_line_atomic loop

state persistence:
  - state.json holds stage per pair under "workflow_states" key
  - restored on watchdog restart (idempotent: GEN_* re-runs are safe; broad
    overwrites configs but writes same number of queue lines)
"""

from enum import Enum
from pathlib import Path
from typing import Optional


class Stage(Enum):
    INIT          = "init"
    GEN_BROAD     = "gen_broad"
    WAIT_BROAD    = "wait_broad"
    GEN_REFINED   = "gen_refined"
    WAIT_REFINED  = "wait_refined"
    GEN_HOLDOUT   = "gen_holdout"
    WAIT_HOLDOUT  = "wait_holdout"
    GEN_PERSIST   = "gen_persist"
    DONE          = "done"
    ERROR         = "error"


class Pair:
    """one (method, experiment) state machine.

    field invariants:
      method, experiment, output_dir, budget, seed: immutable inputs
      stage: current state, advanced by tick()
      error: last exception text on ERROR transition (None otherwise)
      _adapter: lazily resolved (only needed for broad/refined/holdout)
    """

    __slots__ = ("method", "experiment", "output_dir", "budget", "seed",
                 "stage", "error", "_adapter")

    def __init__(self, method: str, experiment: str, output_dir: str,
                 budget: int = 250, seed: int = 1729):
        self.method     = method
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.budget     = budget
        self.seed       = seed
        self.stage      = Stage.INIT
        self.error: Optional[str] = None
        self._adapter   = None

    @property
    def key(self) -> tuple:
        return (self.method, self.experiment)

    def _adapter_or_load(self):
        if self._adapter is None:
            from experiments.utils.hpo.adapters import get_adapter
            self._adapter = get_adapter(self.experiment)
        return self._adapter

    def _count_results(self, stage_name: str) -> int:
        """count FINITE-score trial result JSONs in <output_dir>/<stage>/.

        only counts files where score is a finite number — pre-existing inf or
        NaN trials (e.g., from stale runs) don't falsely satisfy the budget
        gate. silently skips unparseable JSON.
        """
        import json
        import math
        d = self.output_dir / stage_name
        if not d.exists():
            return 0
        n = 0
        for p in d.glob("trial_*.json"):
            stem = p.stem.removeprefix("trial_")
            if not stem.isdigit():
                continue
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            score = data.get("score")
            if isinstance(score, (int, float)) and math.isfinite(score):
                n += 1
        return n

    def tick(self, queue_file: Path) -> bool:
        """advance at most one stage. returns True iff state changed.

        GEN_* stages call workflow.<stage>() which writes configs + sbatch
        lines to queue_file. WAIT_* stages only check disk; no I/O if not
        ready. ERROR stage is sticky (never advanced).
        """
        from experiments.utils.hpo import workflow as wf
        from experiments.utils.hpo.budget import stage_budget
        old = self.stage
        try:
            if self.stage == Stage.INIT:
                wf.recalibrate(self.method, self.experiment, self.output_dir)
                self.stage = Stage.GEN_BROAD

            elif self.stage == Stage.GEN_BROAD:
                wf.broad(self.method, self.experiment, self._adapter_or_load(),
                         n=stage_budget("broad", method=self.method), seed=self.seed,
                         output_dir=self.output_dir, queue_file=queue_file)
                self.stage = Stage.WAIT_BROAD

            elif self.stage == Stage.WAIT_BROAD:
                if self._count_results("broad") >= stage_budget("broad", method=self.method):
                    self.stage = Stage.GEN_REFINED

            elif self.stage == Stage.GEN_REFINED:
                result = wf.refined(self.method, self.experiment,
                                    self._adapter_or_load(),
                                    n=stage_budget("refined", method=self.method), seed=self.seed,
                                    output_dir=self.output_dir,
                                    queue_file=queue_file)
                if result is None or result.get("skipped"):
                    self.stage = Stage.GEN_HOLDOUT
                else:
                    self.stage = Stage.WAIT_REFINED

            elif self.stage == Stage.WAIT_REFINED:
                if self._count_results("refined") >= stage_budget("refined", method=self.method):
                    self.stage = Stage.GEN_HOLDOUT

            elif self.stage == Stage.GEN_HOLDOUT:
                wf.holdout(self.method, self.experiment, self._adapter_or_load(),
                           output_dir=self.output_dir, queue_file=queue_file)
                self.stage = Stage.WAIT_HOLDOUT

            elif self.stage == Stage.WAIT_HOLDOUT:
                if self._count_results("holdout") >= 1:
                    self.stage = Stage.GEN_PERSIST

            elif self.stage == Stage.GEN_PERSIST:
                holdout_files = sorted(
                    (self.output_dir / "holdout").glob("trial_*.json"))
                if holdout_files:
                    wf.persist(self.method, self.experiment, holdout_files[0],
                               self.output_dir)
                self.stage = Stage.DONE

            elif self.stage in (Stage.DONE, Stage.ERROR):
                pass  # terminal

        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"[:400]
            self.stage = Stage.ERROR

        return self.stage != old


def load_pairs(spec_file: Path) -> list:
    """parse workflow-pairs JSON: list of {method, experiment, output_dir,
    budget?, seed?} objects.
    """
    import json
    data = json.loads(Path(spec_file).read_text())
    return [Pair(**d) for d in data]


def serialize_states(pairs: list) -> list:
    """build JSON-safe list of {method, experiment, stage, error} for state.json."""
    return [
        {"method": p.method, "experiment": p.experiment,
         "stage": p.stage.value, "error": p.error}
        for p in pairs
    ]


def restore_states(pairs: list, prior: list) -> None:
    """restore stage from prior serialize_states() output; mutates pairs."""
    by_key = {(d["method"], d["experiment"]): d for d in prior}
    for p in pairs:
        d = by_key.get(p.key)
        if d is None:
            continue
        try:
            p.stage = Stage(d["stage"])
            p.error = d.get("error")
        except ValueError:
            pass  # unknown stage value; leave as INIT
