"""
single source of truth for per-method walltime caps used by watchdog
and pilot-launchers. preempt caps are halved per D3 ratified decision
(gen_tramp.sh:TIME_LIMITS); array caps for reference and non-preempt
usage (submit_hpo.sh:TIME_LIMITS).
"""

import re
from enum import Enum
from typing import Optional


class SpeedClass(Enum):
    """speed rank for queue sorting and cpu eligibility.

    SLOW (0): gpu-only, long runtimes (flow integration), front of queue.
    MEDIUM (1): small nn, mid runtimes (~3-10 min per cpu trial), mid queue.
    FAST (2): cheap methods, short runtimes (~3-5 min per cpu trial), back of queue.
    """
    SLOW   = 0
    MEDIUM = 1
    FAST   = 2


SPEED_CLASS_MAP: dict[str, SpeedClass] = {
    # SLOW (cpu-ineligible): 7 canonical + 1 alias = 8 entries
    "VFM":                       SpeedClass.SLOW,
    "FMDRE":                     SpeedClass.SLOW,
    "FMDRE_S2":                  SpeedClass.SLOW,
    "TriangularFMDRE":           SpeedClass.SLOW,
    "TriangularVFM_V1":          SpeedClass.SLOW,
    "TriangularVFM_V2":          SpeedClass.SLOW,
    "TriangularVFM_V3":          SpeedClass.SLOW,
    "TriangularVFM":             SpeedClass.SLOW,  # legacy alias for TriangularVFM_V1

    # MEDIUM (cpu-eligible): 7 canonical + 1 alias = 8 entries
    "TSM":                       SpeedClass.MEDIUM,
    "CTSM":                      SpeedClass.MEDIUM,
    "BDRE":                      SpeedClass.MEDIUM,
    "TriangularTSM":             SpeedClass.MEDIUM,
    "TriangularCTSM_V1":         SpeedClass.MEDIUM,
    "TriangularCTSM_V2":         SpeedClass.MEDIUM,
    "TriangularCTSM_V3":         SpeedClass.MEDIUM,
    "TriangularCTSM":            SpeedClass.MEDIUM,  # legacy alias for TriangularCTSM_V1

    # FAST (cpu-eligible): 6 canonical + 3 aliases = 9 entries
    "MDRE_15":                   SpeedClass.FAST,
    "TDRE_5":                    SpeedClass.FAST,
    "MultiHeadTriangularTDRE":   SpeedClass.FAST,
    "TriangularMDRE":            SpeedClass.FAST,
    "TabularPluginDRE":          SpeedClass.FAST,
    "SmoothedTabularPluginDRE":  SpeedClass.FAST,
    "MDRE":                      SpeedClass.FAST,  # legacy alias for MDRE_15
    "TDRE":                      SpeedClass.FAST,  # legacy alias for TDRE_5
    "MHTTDRE":                   SpeedClass.FAST,  # legacy alias for MultiHeadTriangularTDRE
}


def speed_rank(method: Optional[str]) -> int:
    """return 0|1|2 (slow|medium|fast). unknown/None/empty -> 1 (neutral).

    used to sort queue at write time: front=slow (0), back=fast (2).
    alias names are looked up directly in SPEED_CLASS_MAP (safe).
    """
    if not method or method not in SPEED_CLASS_MAP:
        return SpeedClass.MEDIUM.value
    return SPEED_CLASS_MAP[method].value


def cpu_eligible(method: Optional[str]) -> bool:
    """true iff method is cpu-eligible (MEDIUM or FAST).
    unknown/None/empty -> False (conservative).

    guards against scheduling gpu-only methods on cpu array.
    """
    if not method or method not in SPEED_CLASS_MAP:
        return False
    return SPEED_CLASS_MAP[method] != SpeedClass.SLOW


def cpu_eligible_methods() -> set[str]:
    """canonical method names eligible for cpu (no aliases, no SLOW).

    derived from intersection of SPEED_CLASS_MAP keys and METHOD_SPECS keys.
    aliases (MDRE, TDRE, TriangularCTSM, TriangularVFM, MHTTDRE) are excluded
    because they don't appear in METHOD_SPECS.keys(); callers should use only
    canonical names in method_filter.

    expected 13 entries: 7 MEDIUM + 6 FAST.
    """
    from experiments.utils.hpo.method_specs import METHOD_SPECS
    return {m for m in SPEED_CLASS_MAP
            if m in METHOD_SPECS and cpu_eligible(m)}


# 4-cell base preempt caps (recalibrated/broad-sweep convention).
# values are 4x-10x observed 4-cell median elapsed (tonight + historical 100 trials).
# tighter caps -> higher slurm scheduling priority on preempt with FairShare+QOS.
# entries cover canonical method names + legacy aliases (MDRE, TDRE, MHTTDRE).
WALLTIME_CAPS_PREEMPT = {
    "TSM":                       "0:05:00",  # observed 4-cell 15s
    "CTSM":                      "0:05:00",
    "VFM":                       "1:15:00",  # int_steps capped at 3000; covers heaviest path at p99 rate
    "BDRE":                      "0:05:00",
    "MDRE":                      "0:05:00",
    "MDRE_15":                   "0:05:00",
    "TDRE":                      "0:05:00",
    "TDRE_5":                    "0:05:00",
    "MHTTDRE":                   "0:05:00",  # observed 4-cell 14s
    "MultiHeadTriangularTDRE":   "0:05:00",  # canonical alias
    "TriangularMDRE":            "0:05:00",
    "FMDRE":                     "1:45:00",  # int_steps capped at 3000; 3 flow integrations -> 1.5x VFM cap
    "FMDRE_S2":                  "1:45:00",
    "TriangularFMDRE":           "1:45:00",
    "TriangularTSM":             "0:10:00",
    "TriangularCTSM_V1":         "0:05:00",  # historical 4-cell 10s
    "TriangularCTSM":            "0:05:00",  # legacy alias for V1
    "TriangularCTSM_V2":         "0:05:00",
    "TriangularCTSM_V3":         "0:10:00",  # heavier 2D path
    "TriangularVFM_V1":          "1:15:00",  # int_steps capped at 3000; covers V3 (heaviest) at p99 rate
    "TriangularVFM":             "1:15:00",  # legacy alias for V1
    "TriangularVFM_V2":          "1:15:00",
    "TriangularVFM_V3":          "1:15:00",  # heaviest path; 3000 steps * 1.35 s/step ~= 67 min
    "TabularPluginDRE":          "0:05:00",
    "SmoothedTabularPluginDRE":  "0:05:00",
}

# array-partition (full, not halved). 2x preempt for slack on non-preemptible jobs.
WALLTIME_CAPS_ARRAY = {k: f"{int(_h)*2}:{_m:02d}:{_s:02d}"
                      for k, v in WALLTIME_CAPS_PREEMPT.items()
                      for _h, _m, _s in [v.split(":")]
                      for _h, _m, _s in [(int(_h), int(_m), int(_s))]}

# cpu-specific per-trial caps. keys must be a subset of cpu_eligible_methods().
# do not add SLOW methods (gpu-ineligible); do not add aliases (use canonical).
# calibrated from pilot 7638426 (CTSM_V1: p50=265s ~4.4 min, p90=559s ~9.3 min).
# extrapolated for related methods; validate before fleet rollout.
WALLTIME_CAPS_CPU = {
    # MEDIUM: small NN training; ~5-10 min per trial on cpu
    "TSM":                       "0:10:00",
    "CTSM":                      "0:10:00",
    "BDRE":                      "0:10:00",
    "TriangularTSM":             "0:10:00",
    "TriangularCTSM_V1":         "0:10:00",
    "TriangularCTSM_V2":         "0:10:00",
    "TriangularCTSM_V3":         "0:12:00",  # heavier 2D path; ~20% slower than V1/V2

    # FAST: small models, expected ~3-5 min per trial on cpu
    "MDRE_15":                   "0:05:00",
    "TDRE_5":                    "0:05:00",
    "MultiHeadTriangularTDRE":   "0:05:00",
    "TriangularMDRE":            "0:05:00",

    # TABULAR: cpu-only methods, expected fast (no neural training)
    "TabularPluginDRE":          "0:05:00",
    "SmoothedTabularPluginDRE":  "0:05:00",
}

CPU_STARTUP_BUFFER_SECONDS = 60        # conda activate + python import + cell preload
CPU_WALLTIME_MAX_SECONDS = 28800       # 8:00:00, matches _from_seconds clamp at line 80; array partition allows 12h but helper caps at 8h

WALLTIME_CAPS = WALLTIME_CAPS_PREEMPT

WALLTIME_DEFAULT_PREEMPT = "0:30:00"
WALLTIME_DEFAULT_ARRAY = "1:00:00"

# random_a/random_b/holdout pilots evaluate 32 cells vs the 4-cell base.
# nominal scaling is 8x; 10x leaves headroom for GPU contention + startup overhead.
RANDOM_SAMPLE_MULT = 10.0


def _to_seconds(hms: str) -> int:
    """parse HH:MM:SS string to seconds.

    args: hms - string in H:MM:SS or HH:MM:SS format
    returns: total seconds as integer
    """
    match = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})$', hms)
    if not match:
        raise ValueError(f"invalid HH:MM:SS format: {hms}")
    h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return h * 3600 + m * 60 + s


def _from_seconds(s: int) -> str:
    """convert seconds to HH:MM:SS string with hard ceiling at 8:00:00.

    args: s - total seconds as integer
    returns: formatted string H:MM:SS or HH:MM:SS (leading zero iff hour >= 10)
    """
    # apply hard ceiling
    s = min(s, 8 * 3600)

    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60

    return f"{h}:{m:02d}:{sec:02d}"


def cap_for(
    method: str,
    partition: str = "preempt",
    pilot_tag: Optional[str] = None
) -> str:
    """lookup walltime cap for method, optionally scaled by pilot_tag.

    args:
      method - method name (e.g. "TSM", "FMDRE")
      partition - "preempt" (default) or "array"
      pilot_tag - optional, one of {None, "random_a", "random_b", "optuna", "recalibrated"}

    logic:
      1. select dict based on partition
      2. lookup base_cap from dict with fallback to default
      3. if pilot_tag in {"random_a", "random_b"}:
         - parse base_cap as HH:MM:SS -> seconds
         - multiply by RANDOM_SAMPLE_MULT
         - round up to nearest minute
         - convert back to HH:MM:SS
      4. cap result at hard ceiling "8:00:00"
      5. return as H:MM:SS or HH:MM:SS string

    returns: walltime cap as string in H:MM:SS or HH:MM:SS format
    """
    # select dict based on partition
    if partition == "preempt":
        caps_dict = WALLTIME_CAPS_PREEMPT
        default_cap = WALLTIME_DEFAULT_PREEMPT
    elif partition == "array":
        caps_dict = WALLTIME_CAPS_ARRAY
        default_cap = WALLTIME_DEFAULT_ARRAY
    else:
        raise ValueError(f"unknown partition: {partition}")

    # lookup base cap
    base_cap = caps_dict.get(method, default_cap)

    # apply random sample multiplier if needed
    if pilot_tag in {"random_a", "random_b"}:
        secs = _to_seconds(base_cap)
        mult_secs = secs * RANDOM_SAMPLE_MULT
        rounded_secs = int((mult_secs + 59) / 60) * 60
        return _from_seconds(rounded_secs)

    return base_cap


def cpu_cap_for(method: str, n_per_element: int = 8,
                safety_factor: float = 1.3) -> str:
    """compute per-element walltime: per_trial_cap × n_per_element × safety + startup.

    raises ValueError if method not in WALLTIME_CAPS_CPU (cpu-ineligible).
    safety_factor < 1.0 is clamped to 1.0. result is clamped to 12:00:00.

    args:
        method: canonical method name (must be in WALLTIME_CAPS_CPU).
        n_per_element: number of trials per array element (default 8).
        safety_factor: multiplicative headroom (default 1.3, clamped to >= 1.0).

    returns:
        walltime cap as "H:MM:SS" or "HH:MM:SS".

    raises:
        ValueError: if method not in WALLTIME_CAPS_CPU.
    """
    if method not in WALLTIME_CAPS_CPU:
        raise ValueError(f"{method} not in WALLTIME_CAPS_CPU or cpu-ineligible")

    safety_factor = max(safety_factor, 1.0)
    per_trial_sec = _to_seconds(WALLTIME_CAPS_CPU[method])
    total_sec = int(per_trial_sec * n_per_element * safety_factor) + CPU_STARTUP_BUFFER_SECONDS
    total_sec = min(total_sec, CPU_WALLTIME_MAX_SECONDS)

    return _from_seconds(total_sec)


def compute_element_walltime(method_filter: list[str],
                             n_per_element: int = 8,
                             safety_factor: float = 1.3) -> str:
    """take MAX over method_filter's per-method cpu caps, scale to element walltime.

    used by cpu_dispatcher to compute required walltime when --walltime=auto.
    takes conservative approach: use the slowest (max) method's cap in filter.

    args:
        method_filter: list of canonical method names (all must be cpu-eligible).
        n_per_element: number of trials per array element (default 8).
        safety_factor: multiplicative headroom (default 1.3, clamped to >= 1.0).

    returns:
        walltime cap as "H:MM:SS" or "HH:MM:SS".

    raises:
        ValueError: if method_filter is empty OR any method is cpu-ineligible.
    """
    if not method_filter:
        raise ValueError("method_filter is empty")

    bad = [m for m in method_filter if m not in WALLTIME_CAPS_CPU]
    if bad:
        raise ValueError(f"cpu-ineligible methods in filter: {bad}")

    max_cap_sec = max(_to_seconds(WALLTIME_CAPS_CPU[m]) for m in method_filter)
    safety_factor = max(safety_factor, 1.0)
    total_sec = int(max_cap_sec * n_per_element * safety_factor) + CPU_STARTUP_BUFFER_SECONDS
    total_sec = min(total_sec, CPU_WALLTIME_MAX_SECONDS)

    return _from_seconds(total_sec)
