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

resource roots are taken directly from the environment -- no path is invented
here. cluster setup (e.g. ~/.bashrc) must export them; a missing var is a hard
error, never a silent $HOME or cluster-hardcoded fallback:
    DPE_DATA_ROOT -> NFS-shared data root
    DPE_CKPT_ROOT -> node-local scratch ckpt root
"""
import os

import yaml


for _var in ("DPE_DATA_ROOT", "DPE_CKPT_ROOT"):
    if not os.environ.get(_var):
        raise RuntimeError(
            f"{_var} is not set -- cluster setup must export it before "
            f"importing ex.* (see ~/.bashrc)."
        )


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
