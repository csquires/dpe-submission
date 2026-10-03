"""safe frozen-fit eldr model builder.

thin shared helper for per-trial model fitting across experiments.
wraps METHOD_SPECS builders with:
  - alias resolution (canonical name lookup via ALIAS_PAIRS)
  - legacy hyperparameter key normalization (n_epochs -> n_steps, per method)
  - safe fit/predict (never raises; returns FitResult)
  - optional seeding (torch + numpy, before build)

does NOT reimplement load_winners/resolve_hp/detect_schema (load_winners.py)
or _atomic_h5_write (worker.py); callers import those separately.
"""

from collections import namedtuple
import math

import numpy as np
import torch

from ex.utils.hpo.method_specs import ALIAS_PAIRS, METHOD_SPECS


FitResult = namedtuple("FitResult", ["ok", "value", "error"])
"""ok: bool (True if fit succeeded and value is finite)
value: float | None (scalar eldr estimate, or None if ok=False)
error: str | None (error tag if ok=False; None if ok=True)
"""


METHOD_ALIAS = {p[0]: p[1] for p in ALIAS_PAIRS}
"""canonical name lookup: short name -> canonical name.
e.g. {"MHTTDRE": "MultiHeadTriangularTDRE", "MDRE": "MDRE_15", ...}
"""


_HP_KEY_ALIAS_BY_METHOD = {
    "VFM":           {"n_epochs": "n_steps"},
    "VFMOrthros":    {"n_epochs": "n_steps"},
    "CTSM":          {"n_epochs": "n_steps"},
    "FMDRE":         {"n_epochs": "n_steps"},
    "FMDRE_S2":      {"n_epochs": "n_steps"},
}
"""per-method legacy key aliases. builders expect canonical names."""


def normalize_hp(method: str, hp: dict) -> dict:
    """remap legacy hyperparameter keys to canonical names.

    procedure:
      1. look up _HP_KEY_ALIAS_BY_METHOD.get(method, {}) -> alias dict
      2. for each (k, v) in hp.items():
           canonical_k = alias.get(k, k)
           yield canonical_k: v
      3. return new dict

    example:
      normalize_hp("VFM", {"n_epochs": 100, "lr": 1e-3})
        -> {"n_steps": 100, "lr": 1e-3}
      normalize_hp("BDRE", {"n_steps": 100, "lr": 1e-3})
        -> {"n_steps": 100, "lr": 1e-3}  (unchanged)
    """
    alias = _HP_KEY_ALIAS_BY_METHOD.get(method, {})
    return {alias.get(k, k): v for k, v in hp.items()}


def fit_eldr(method, hp, joint, shuffled, *, input_dim, device, seed=None):
    """safe fit + predict eldr with error handling.

    procedure:
      1. resolve alias -> canonical name:
           canonical = METHOD_ALIAS.get(method, method)
         if canonical not in METHOD_SPECS:
           raise KeyError (setup error; do NOT catch)

      2. extract builder + metadata from METHOD_SPECS[canonical]:
           spec = METHOD_SPECS[canonical]
           builder = spec["builder"]
           requires_pstar = spec.get("requires_pstar", False)
           num_waypoints = spec.get("num_waypoints", None)

      3. optional seeding (if seed is not None):
           torch.manual_seed(seed)
           np.random.seed(seed)

      4. build model:
           builder_kwargs = {
             "input_dim": input_dim,
             "device": device,
             "num_waypoints": num_waypoints if num_waypoints is not None else 0,
             **normalize_hp(canonical, hp),
           }
           try:
             model = builder(**builder_kwargs)
           except Exception as e:
             (single block, handle OOM + other exceptions)

      5. fit model:
           try:
             if requires_pstar:
               model.fit(joint, shuffled, joint)
             else:
               model.fit(joint, shuffled)
           except Exception as e:
             (single block)

      6. predict eldr:
           try:
             with torch.no_grad():
               val = float(model.predict_eldr(joint).item())
           except Exception as e:
             (single block)

      7. guard non-finite (SEPARATE from exception handling):
           if not math.isfinite(val):
             return FitResult(False, None, "non_finite")

      8. success:
           return FitResult(True, val, None)

    returns FitResult; never raises (all errors wrapped).
    KeyError on canonical lookup is a setup error and propagates.
    """
    # step 1: resolve alias
    canonical = METHOD_ALIAS.get(method, method)
    if canonical not in METHOD_SPECS:
        raise KeyError(f"method {method} (canonical: {canonical}) not in METHOD_SPECS")

    # step 2: extract spec
    spec = METHOD_SPECS[canonical]
    builder = spec["builder"]
    requires_pstar = spec.get("requires_pstar", False)
    num_waypoints = spec.get("num_waypoints", None)

    # step 3: optional seeding
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # step 4: build model (single except block)
    builder_kwargs = {
        "input_dim": input_dim,
        "device": device,
        "num_waypoints": num_waypoints if num_waypoints is not None else 0,
        **normalize_hp(canonical, hp),
    }
    try:
        model = builder(**builder_kwargs)
    except Exception as e:
        if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return FitResult(False, None, "cuda_oom")
        else:
            return FitResult(False, None, f"exception:{type(e).__name__}")

    # step 5: fit model (single except block)
    try:
        if requires_pstar:
            model.fit(joint, shuffled, joint)
        else:
            model.fit(joint, shuffled)
    except Exception as e:
        if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return FitResult(False, None, "cuda_oom")
        else:
            return FitResult(False, None, f"exception:{type(e).__name__}")

    # step 6: predict eldr (single except block)
    try:
        with torch.no_grad():
            val = float(model.predict_eldr(joint).item())
    except Exception as e:
        if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return FitResult(False, None, "cuda_oom")
        else:
            return FitResult(False, None, f"exception:{type(e).__name__}")

    # step 7: guard non-finite (separate check after successful predict)
    if not math.isfinite(val):
        return FitResult(False, None, "non_finite")

    # step 8: success
    return FitResult(True, val, None)
