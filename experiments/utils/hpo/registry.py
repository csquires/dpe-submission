from copy import deepcopy

from experiments.utils.hpo.method_specs import METHOD_SPECS


LEGACY_ALIASES: dict[str, str] = {
    "MHTTDRE": "MultiHeadTriangularTDRE",
    "MDRE": "MDRE_15",
    "TDRE": "TDRE_5",
    "TriangularCTSM": "TriangularCTSM_V1",
    "TriangularVFM": "TriangularVFM_V1",
}


def build_search_spaces(
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    overrides: dict[str, dict] | None = None,
    include_tabular: bool = False,
) -> dict[str, dict]:
    """
    build canonical SEARCH_SPACES dict from METHOD_SPECS with filtering and overrides.

    each entry in result holds {search_space, builder, requires_pstar, num_waypoints}
    where search_space is param -> spec tuple and num_waypoints is copied from
    METHOD_SPECS[method]. legacy alias keys are injected so both names reference
    the same dict object (shared reference, not copy).

    args:
        include: list of canonical method names to include. if None, includes all
            continuous (non-tabular) methods. if supplied, other methods excluded.
        exclude: list of canonical method names to skip after include is applied.
            no-op if name not in include set.
        overrides: dict mapping method name -> {param: new_spec_tuple}. updates
            search_space[param] for matched method. raises KeyError if method
            not in result.
        include_tabular: if True, also include TabularPluginDRE and
            SmoothedTabularPluginDRE. default False (continuous only).

    returns:
        dict with all canonical method names + 5 legacy alias keys. both names
        point to same entry (shared reference). ready for argparse choices=keys().

    raises:
        KeyError: if override targets a method not in result dict.
    """
    # resolve canonical method set
    tabular_methods = {"TabularPluginDRE", "SmoothedTabularPluginDRE"}
    all_methods = set(METHOD_SPECS.keys())

    # filter to continuous if not include_tabular
    if not include_tabular:
        canonical_set = all_methods - tabular_methods
    else:
        canonical_set = all_methods

    # apply include filter
    if include is not None:
        # resolve alias names to canonical
        resolved_include = set()
        for name in include:
            if name in LEGACY_ALIASES:
                resolved_include.add(LEGACY_ALIASES[name])
            else:
                resolved_include.add(name)
        canonical_set = canonical_set & resolved_include

    # apply exclude filter
    if exclude is not None:
        canonical_set = canonical_set - set(exclude)

    # build per-method entries
    result = {}
    for method in canonical_set:
        base_spec = METHOD_SPECS[method]
        entry = {
            "search_space": deepcopy(base_spec["base_search_space"]),
            "builder": base_spec["builder"],
            "requires_pstar": base_spec["requires_pstar"],
            "num_waypoints": base_spec.get("num_waypoints", None),
        }
        result[method] = entry

    # apply overrides
    if overrides is not None:
        for method, param_updates in overrides.items():
            if method not in result:
                raise KeyError(f"method '{method}' not in SEARCH_SPACES after filtering")
            result[method]["search_space"].update(param_updates)

    # inject legacy aliases
    for old_name, canonical_name in LEGACY_ALIASES.items():
        if canonical_name in result:
            result[old_name] = result[canonical_name]

    return result
