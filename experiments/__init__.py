"""experiments package init.

guards against the dominant leak vector that fills $HOME on this cluster:
config yamls with relative paths like "experiments/<exp>/data" that resolve
under the repo's CWD (which is typically /home/<user>/dpe-submission).

procedure executed once on import (when the user runs
`python -m experiments.<exp>.<step>` for any step):
    1. install DPE_DATA_ROOT and DPE_CKPT_ROOT env-var defaults if unset.
    2. monkey-patch yaml.safe_load and yaml.load to recursively
       os.path.expandvars on every string in the parsed structure.

callers that already use src.utils.io._load_config get the same expansion via
that helper. callers that do `yaml.safe_load(open(...))` directly get it
through the patch installed here. all configs should template path keys as
${DPE_DATA_ROOT}/<exp>/... or ${DPE_CKPT_ROOT}/<exp>/ckpt.

defaults:
    DPE_DATA_ROOT -> /data/user_data/$USER/dpe-submission     (NFS, all nodes)
    DPE_CKPT_ROOT -> /scratch/$USER/ckpt/dpe-submission       (node-local, fast)
"""
import os
import getpass

import yaml


_USER = os.environ.get("USER") or getpass.getuser()
os.environ.setdefault("DPE_DATA_ROOT", f"/data/user_data/{_USER}/dpe-submission")
os.environ.setdefault("DPE_CKPT_ROOT", f"/scratch/{_USER}/ckpt/dpe-submission")


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
