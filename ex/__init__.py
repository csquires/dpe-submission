"""experiments package init.

guards against the dominant leak vector that fills $HOME on shared clusters:
config yamls with relative paths like "ex/<exp>/data" that resolve under the
repo's CWD.

procedure executed once on import (when the user runs
`python -m ex.<exp>.<step>` for any step):
    1. install DPE_DATA_ROOT and DPE_CKPT_ROOT env-var defaults if unset.
    2. monkey-patch yaml.safe_load and yaml.load to recursively
       os.path.expandvars on every string in the parsed structure.

callers that already use src.utils.io._load_config get the same expansion via
that helper. callers that do `yaml.safe_load(open(...))` directly get it
through the patch installed here. all configs should template path keys as
${DPE_DATA_ROOT}/<exp>/... or ${DPE_CKPT_ROOT}/<exp>/ckpt.

defaults (override by exporting the env var before running):
    DPE_DATA_ROOT -> $HOME/dpe-data        (set NFS path on clusters)
    DPE_CKPT_ROOT -> $HOME/dpe-ckpt        (set node-local scratch on clusters)
"""
import os

import yaml


os.environ.setdefault("DPE_DATA_ROOT", os.path.expanduser("~/dpe-data"))
os.environ.setdefault("DPE_CKPT_ROOT", os.path.expanduser("~/dpe-ckpt"))


def _expand_env(value):
    """recursively os.path.expandvars on every string in a parsed yaml structure."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


# idempotency guard: src.utils.io may patch first; respect the sentinel.
if not getattr(yaml, "_dpe_envvar_patched", False):
    _orig_safe_load = yaml.safe_load
    _orig_load = yaml.load

    def _patched_safe_load(stream):
        return _expand_env(_orig_safe_load(stream))

    def _patched_load(stream, *args, **kwargs):
        return _expand_env(_orig_load(stream, *args, **kwargs))

    yaml.safe_load = _patched_safe_load
    yaml.load = _patched_load
    yaml._dpe_envvar_patched = True
