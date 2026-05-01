"""
single source of truth for per-method walltime caps used by watchdog
and pilot-launchers. preempt caps are halved per D3 ratified decision
(gen_tramp.sh:TIME_LIMITS); array caps for reference and non-preempt
usage (submit_hpo.sh:TIME_LIMITS).
"""

import re
from typing import Optional


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
