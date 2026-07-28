"""build ms_ref/tr8192_te{2048,4096}.h5 from model_selection processed results,
matching the existing tr8192_te8192.h5 key format so plot_vs_nstar's load_ms reads
them unchanged.

source: ex/synth/model_selection/processed_results/tr8192_te{K}/new_pstar.h5
dest:   ex/ablations/dokls/ms_ref/tr8192_te{K}.h5   for K in {2048, 4096}

usage: python -m ex.ablations.dokls.make_ms_ref_nstar
"""
import os
from pathlib import Path

import h5py


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # ex/ablations/dokls -> repo root
MS_PROC = REPO / "ex" / "synth" / "model_selection" / "processed_results"
MS_REF = HERE / "ms_ref"
TESTS = [2048, 4096]                          # 8192 ref already exists
DROP_PREFIX = "regret_"                       # sole difference vs the source


def build_ref(K):
    """copy tr8192_te{K}/new_pstar.h5 -> ms_ref/tr8192_te{K}.h5, dropping regret_*.

    returns (dst_path, n_copied, n_dropped).
    """
    src = MS_PROC / f"tr8192_te{K}" / "new_pstar.h5"
    dst = MS_REF / f"tr8192_te{K}.h5"
    if not src.exists():
        raise FileNotFoundError(f"source missing: {src}")
    MS_REF.mkdir(parents=True, exist_ok=True)

    n_copy = n_drop = 0
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for k in fin.keys():
            if k.startswith(DROP_PREFIX):
                n_drop += 1
                continue
            d = fout.create_dataset(k, data=fin[k][:])
            for a, v in fin[k].attrs.items():
                d.attrs[a] = v
            n_copy += 1
    return dst, n_copy, n_drop


def diff_against_reference(built_paths):
    """diff each built ref's key set against the existing tr8192_te8192.h5 ref."""
    ref8 = MS_REF / "tr8192_te8192.h5"
    if not ref8.exists():
        print(f"  [warn] reference {ref8} absent; skipping key diff")
        return
    with h5py.File(ref8, "r") as f:
        ref_keys = set(f.keys())
    for dst in built_paths:
        with h5py.File(dst, "r") as f:
            keys = set(f.keys())
        extra, missing = keys - ref_keys, ref_keys - keys
        ok = not extra and not missing
        print(f"  {dst.name}: {len(keys)} keys, key-set == te8192 ref: {ok}"
              + ("" if ok else f" (extra={sorted(extra)}, missing={sorted(missing)})"))


def main():
    built = []
    for K in TESTS:
        dst, n_copy, n_drop = build_ref(K)
        print(f"wrote {dst}  ({n_copy} copied, {n_drop} regret_* dropped)")
        built.append(dst)
    print("key-diff vs existing ms_ref/tr8192_te8192.h5:")
    diff_against_reference(built)


if __name__ == "__main__":
    main()
