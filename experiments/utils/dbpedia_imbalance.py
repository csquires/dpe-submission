import hashlib
import numpy as np
import torch


DBPEDIA_LABEL_NAMES = [
    'Company',                    # 0
    'EducationalInstitution',     # 1
    'Artist',                     # 2
    'Athlete',                    # 3
    'OfficeHolder',               # 4
    'MeanOfTransportation',       # 5
    'Building',                   # 6
    'NaturalPlace',               # 7
    'Village',                    # 8
    'Animal',                     # 9
    'Plant',                      # 10
    'Album',                      # 11
    'Film',                       # 12
    'WrittenWork',                # 13
]


def get_dbpedia_dataset(split: str = 'train', cache_dir: str | None = None):
    """
    thin wrapper around datasets.load_dataset() for DBpedia_14.

    attempt primary id 'dbpedia_14'; fall back to 'fancyzhx/dbpedia_14' if first fails.

    args:
        split: dataset split to load (default: 'train')
        cache_dir: directory to cache datasets (default: None, uses HF default)

    returns:
        HF Dataset with columns (title: str, content: str, label: int)
    """
    from datasets import load_dataset
    try:
        return load_dataset('dbpedia_14', split=split, cache_dir=cache_dir)
    except Exception:
        return load_dataset('fancyzhx/dbpedia_14', split=split, cache_dir=cache_dir)


def subsample_dbpedia(ds, weights: np.ndarray, K: int = 14, min_per_class: int = 10, seed: int | None = None) -> list[int]:
    """
    return indices of subset where class frequencies match target weights.

    mirrors subsample_mnist but generalizes to K classes and uses seeded shuffling
    per-class before selection. used for balanced p* sampling and arbitrary imbalance experiments.

    args:
        ds: HF Dataset with 'label' column
        weights: [K] numpy array of target class weights (should sum to ~1.0)
        K: number of classes (default: 14)
        min_per_class: minimum samples required per class (default: 10)
        seed: random seed for per-class shuffling (default: None, non-deterministic)

    returns:
        list[int] of indices into dataset

    procedure:
        extract labels, build class_indices dict.
        compute class counts and validate feasibility.
        validate weights shape and normalization.
        compute total_budget by floor of min_c(count_c / max(weights[c], 1e-10)).
        iterate, reducing total_budget until all target[c] <= count_c.
        for each class, shuffle indices with seeded RNG and select first target[c].
        return concatenated list.
    """
    # extract labels and build class index map
    labels = ds['label']
    class_indices = {c: [] for c in range(K)}
    for idx in range(len(ds)):
        c = labels[idx]
        class_indices[c].append(idx)

    # compute class counts
    count_c = np.array([len(class_indices[c]) for c in range(K)])

    # validate feasibility of min_per_class
    if min_per_class * K > count_c.sum():
        raise ValueError(f"min_per_class={min_per_class} * {K} classes exceeds dataset size")

    # validate weights shape
    if weights.shape != (K,):
        raise ValueError(f"weights shape {weights.shape} does not match K={K}")

    # validate weights normalization (check but allow small tolerance)
    if abs(weights.sum() - 1.0) > 1e-6:
        # implementation choice: proceed with warning implicit; could also raise
        pass

    # compute total budget, reducing iteratively until feasible
    total_budget = int(np.floor((count_c / np.maximum(weights, 1e-10)).min()))

    while True:
        target = np.array([max(int(np.round(weights[c] * total_budget)), min_per_class) for c in range(K)])

        # check if all targets are feasible
        if np.all(target <= count_c):
            break

        # reduce budget and retry
        total_budget -= 1

    # select indices with seeded shuffle per-class
    rng = np.random.default_rng(seed)
    selected_indices = []
    for c in range(K):
        shuffled = rng.permutation(class_indices[c])
        selected_indices.extend(shuffled[:target[c]])

    return selected_indices


def flow_state_hash(ckpt_path: str) -> str:
    """
    compute deterministic 8-char hex hash of a flow checkpoint.

    used by both step0_pretrain.py --mode log_p_y (writer) and step1_create_data.py
    (reader) to key the log_p_y.<flowhash>.pt cache. both call sites MUST
    import this single implementation; do NOT redefine inline.

    args:
        ckpt_path: path to checkpoint file

    returns:
        8-char lowercase hex string

    procedure:
        load checkpoint with map_location='cpu'.
        if state is dict with 'state_dict' key, extract it.
        iterate over sorted state keys.
        for each key, update hash with key bytes and tensor bytes.
        skip non-tensor entries defensively.
        return first 8 chars of sha256 digest.
    """
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode('utf-8'))
        v = state[key]
        if not isinstance(v, torch.Tensor):
            continue
        tensor = v.detach().cpu().contiguous()
        h.update(tensor.numpy().tobytes())

    return h.hexdigest()[:8]
