"""narrow search-space specs around observed top-K values."""

from typing import Tuple, List


def narrow_spec(spec: Tuple, top_k_values: List) -> Tuple:
    """
    narrow a search-space spec tuple around observed top-K values.

    spec: tuple from SEARCH_SPACES (e.g., ("log_uniform", lo, hi))
    top_k_values: list of observed values from top-K trials

    returns: narrowed spec tuple of same kind, with bounds set to [min, max] of top_k_values.
             if min == max, collapses to ("choice", [value]).

    dispatch per spec[0]:
      "log_uniform": ("log_uniform", min(top_k_values), max(top_k_values))
                     → if min == max: ("choice", [min])

      "log_uniform_int": ("log_uniform_int", min(top_k_values), max(top_k_values))
                         → if min == max: ("choice", [int(min)])
                         NOTE: preserves log-uniform behavior; previous version downgraded to uniform_int.

      "uniform": ("uniform", min(top_k_values), max(top_k_values))
                 → if min == max: ("choice", [min])

      "uniform_int": ("uniform_int", min(top_k_values), max(top_k_values))
                     (no min==max collapse; discrete already, min/max are integers)

      "choice": ("choice", sorted(set(top_k_values)))
                (return unique values observed, sorted, as new choice set)

      else: raise ValueError(f"unknown spec type: {spec[0]}")
    """
    spec_type = spec[0]
    # robust to None values mixed with numerics (e.g. ema_decay shifting from
    # 0.999 to None mid-campaign). drop Nones for numeric specs unless they are
    # the only values observed.
    if spec_type != "choice":
        non_none = [v for v in top_k_values if v is not None]
        if non_none:
            top_k_values = non_none
    if not top_k_values or all(v is None for v in top_k_values):
        return spec  # nothing usable; fall back to base spec

    if spec_type == "log_uniform":
        lo, hi = min(top_k_values), max(top_k_values)
        if lo == hi:
            return ("choice", [lo])
        return ("log_uniform", lo, hi)

    elif spec_type == "log_uniform_int":
        lo, hi = min(top_k_values), max(top_k_values)
        if lo == hi:
            return ("choice", [int(lo)])
        return ("log_uniform_int", int(lo), int(hi))

    elif spec_type == "uniform":
        lo, hi = min(top_k_values), max(top_k_values)
        if lo == hi:
            return ("choice", [lo])
        return ("uniform", lo, hi)

    elif spec_type == "uniform_int":
        lo, hi = min(top_k_values), max(top_k_values)
        return ("uniform_int", int(lo), int(hi))

    elif spec_type == "choice":
        # sort key handles None alongside numerics/strings
        uniq = list(set(top_k_values))
        uniq.sort(key=lambda x: (x is None, x if x is not None else 0))
        return ("choice", uniq)

    else:
        raise ValueError(f"unknown spec type: {spec_type}")
