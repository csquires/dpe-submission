"""winners.yaml loader for step2_runner.

supports two yaml schemas:

  schema A (current, used by scratch/200broad/winners_pinned/):
    methods:
      <method>:
        hyperparams: { ... }       # default for all buckets
        score: { ... }             # provenance
        per_bucket:                # optional override per bucket id
          <bucket_id>: { hyperparams: { ... }, ... }
    provenance: { ... }

  schema B (legacy, used by ex/<exp>/winners.yaml):
    <method>:
      <bucket_idx>:
        - hyperparams: { ... }       # top-K list
          mae_median: ...
          trial_id: ...

resolve_hp(yaml_path, method, bucket_id) returns the resolved hp dict for the
given (method, bucket). priority:
  1. schema A:  methods[method].per_bucket[bucket_id].hyperparams (override)
  2. schema A:  methods[method].hyperparams (default)
  3. schema B:  <method>[<bucket>][0].hyperparams (top-1 from list)

raises KeyError if no match found, with a clear message about which keys exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_winners(yaml_path: str | Path) -> dict:
    """load winners.yaml and return raw dict. empty dict on failure."""
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"winners.yaml not found: {p}")
    with open(p) as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"winners.yaml at {p} is not a dict (got {type(d).__name__})")
    return d


def detect_schema(winners: dict) -> str:
    """detect schema A (with top-level 'methods' key) vs schema B (method names at top level)."""
    if "methods" in winners and isinstance(winners["methods"], dict):
        return "A"
    return "B"


def list_methods(winners: dict) -> list[str]:
    """return registered method names regardless of schema."""
    if detect_schema(winners) == "A":
        return list(winners.get("methods", {}).keys())
    return [k for k in winners.keys() if isinstance(winners[k], dict)]


def resolve_hp(winners: dict, method: str, bucket_id: Any = None) -> dict:
    """resolve the hp dict for (method, bucket).

    bucket_id is opaque (str or int); if None, returns the schema-A default
    (no per-bucket lookup). for schema B, falls back to first bucket present
    when bucket_id is None.

    raises KeyError if method or required bucket is missing; the error includes
    the available keys to make debugging easy.
    """
    schema = detect_schema(winners)
    if schema == "A":
        methods = winners.get("methods", {})
        if method not in methods:
            raise KeyError(f"method {method!r} not in winners (schema A); "
                           f"have: {sorted(methods.keys())}")
        m = methods[method]
        if not isinstance(m, dict):
            raise ValueError(f"methods[{method!r}] is not a dict (got {type(m).__name__})")
        if bucket_id is not None:
            per_bucket = m.get("per_bucket", {}) or {}
            if bucket_id in per_bucket:
                entry = per_bucket[bucket_id]
                if isinstance(entry, dict) and "hyperparams" in entry:
                    return dict(entry["hyperparams"])
        if "hyperparams" not in m:
            raise KeyError(f"methods[{method!r}] missing 'hyperparams' key")
        return dict(m["hyperparams"])
    # schema B
    if method not in winners:
        raise KeyError(f"method {method!r} not in winners (schema B); "
                       f"have: {sorted(winners.keys())}")
    buckets = winners[method]
    if not isinstance(buckets, dict):
        raise ValueError(f"winners[{method!r}] is not a dict (got {type(buckets).__name__})")
    if bucket_id is None:
        # fall back to first bucket
        if not buckets:
            raise KeyError(f"winners[{method!r}] has no buckets")
        bucket_id = next(iter(buckets.keys()))
    if bucket_id not in buckets:
        raise KeyError(f"bucket {bucket_id!r} not in winners[{method!r}]; "
                       f"have: {sorted(buckets.keys())}")
    entry = buckets[bucket_id]
    if isinstance(entry, list):
        if not entry:
            raise KeyError(f"winners[{method!r}][{bucket_id!r}] is an empty list")
        first = entry[0]
        if not isinstance(first, dict) or "hyperparams" not in first:
            raise ValueError(f"winners[{method!r}][{bucket_id!r}][0] missing 'hyperparams'")
        return dict(first["hyperparams"])
    if isinstance(entry, dict):
        if "hyperparams" in entry:
            return dict(entry["hyperparams"])
    raise ValueError(f"winners[{method!r}][{bucket_id!r}] has unrecognized format: "
                     f"{type(entry).__name__}")
