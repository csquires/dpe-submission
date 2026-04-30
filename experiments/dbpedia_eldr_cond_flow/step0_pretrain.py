import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from experiments.utils.sbert import encode_corpus, load_or_encode
from experiments.utils.pca import fit_pca_basis, apply_basis
from experiments.utils.dbpedia_imbalance import (
    get_dbpedia_dataset,
    subsample_dbpedia,
    flow_state_hash,
)
from src.models.flow import (
    ClassCondVelocityMLP,
    train_class_cond_flow,
    log_prob_class_cond,
)


def expand_paths(config: dict) -> dict:
    """expand env vars in any string config value.

    iterates keys; for each value v that is a string containing '$',
    replaces with os.path.expandvars(v). mutates config in place;
    returns config.
    """
    for key, value in config.items():
        if isinstance(value, str) and "$" in value:
            config[key] = os.path.expandvars(value)
    return config


def get_device(device_str: str) -> str:
    """validate and fallback device.

    if device_str starts with 'cuda' and torch.cuda.is_available() is False,
    print warning and return 'cpu'. else return device_str as-is.
    """
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("warning: cuda not available, falling back to cpu")
        return "cpu"
    return device_str


def mode_encode(config: dict, device: str, force: bool) -> None:
    """encode DBpedia train split via SBERT; cache embeddings.

    procedure:
        ensure data_dir + hf_cache_dir exist.
        set HF_DATASETS_CACHE env var.
        load dataset.
        extract texts and labels.
        call load_or_encode with cache_path, force, model_name, batch_size, device.
        print summary.
    """
    Path(config["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["hf_cache_dir"]).mkdir(parents=True, exist_ok=True)

    os.environ["HF_DATASETS_CACHE"] = config["hf_cache_dir"]

    ds = get_dbpedia_dataset("train", cache_dir=config["hf_cache_dir"])
    texts = list(ds["content"])
    labels = list(ds["label"])

    # optional smoke-mode subsampling: keep first N (stratified-ish since
    # dbpedia_14 train split is class-blocked, we shuffle first with the
    # config seed to retain class balance).
    n_keep = config.get("corpus_subsample_size")
    if n_keep is not None and n_keep < len(texts):
        rng = np.random.default_rng(config.get("seed", 0))
        idx = rng.permutation(len(texts))[:n_keep]
        texts = [texts[int(i)] for i in idx]
        labels = [labels[int(i)] for i in idx]
        print(f"corpus_subsample_size={n_keep}; using {len(texts)} texts")

    data = load_or_encode(
        texts,
        labels,
        cache_path=f"{config['data_dir']}/embeddings.pt",
        force=force,
        model_name=config["sbert_model"],
        batch_size=config["sbert_batch_size"],
        device=device,
    )

    print(
        f"encoded {len(texts)} texts to shape {tuple(data['embeddings'].shape)}; "
        f"label range [0, {int(data['labels'].max())}]"
    )


def mode_pca(config: dict, force: bool) -> None:
    """fit PCA basis on embeddings; cache basis + standardization stats.

    procedure:
        load embeddings dict.
        check cache; return if exists and not force.
        call fit_pca_basis.
        atomic save to pca_basis.pt.
        print summary.
    """
    data = torch.load(
        f"{config['data_dir']}/embeddings.pt", map_location="cpu"
    , weights_only=False)
    emb_t = data["embeddings"]

    pca_path = f"{config['data_dir']}/pca_basis.pt"
    if Path(pca_path).exists() and not force:
        print(f"pca basis exists at {pca_path}; skipping")
        return

    basis = fit_pca_basis(
        emb_t.numpy(), n_components=config["pca_dim"], seed=config["seed"]
    )

    tmp = f"{pca_path}.tmp"
    torch.save(basis, tmp)
    os.replace(tmp, pca_path)

    print(
        f"PCA basis fit; explained_variance_ratio sum = "
        f"{float(basis['explained_variance_ratio'].sum()):.4f}"
    )


def mode_separability(config: dict, device: str, force: bool) -> None:
    """diagnostic gate—fit linear probe on standardized codes; report separability metrics.

    procedure:
        load embeddings dict and basis.
        apply basis to get codes.
        stratified 80/20 split.
        fit logistic regression.
        evaluate test accuracy.
        compute per-class Gaussians and Bhattacharyya distances.
        generate figure: confusion matrix and Bhattacharyya heatmap.
        log results; warn if test_acc < 0.5 but do NOT block.
    """
    data = torch.load(
        f"{config['data_dir']}/embeddings.pt", map_location="cpu"
    , weights_only=False)
    basis = torch.load(f"{config['data_dir']}/pca_basis.pt", weights_only=False)

    emb_t = data["embeddings"]
    labels_t = data["labels"]

    codes = apply_basis(emb_t, basis).numpy()  # [N, pca_dim]
    labels_np = labels_t.numpy()

    # stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        codes,
        labels_np,
        test_size=0.2,
        stratify=labels_np,
        random_state=config["seed"],
    )

    # fit logistic regression
    clf = LogisticRegression(
        multi_class="multinomial", max_iter=1000, random_state=config["seed"]
    )
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)

    # compute per-class Gaussians and Bhattacharyya distances
    means = []
    covs = []
    for k in range(14):
        codes_k = codes[labels_np == k]
        means.append(codes_k.mean(axis=0))
        covs.append(np.cov(codes_k.T))
    means = np.array(means)  # [14, pca_dim]
    covs = np.array(covs)  # [14, pca_dim, pca_dim]

    # compute 14x14 Bhattacharyya distance matrix
    def bhattacharyya_distance(mu1, cov1, mu2, cov2):
        """compute Bhattacharyya distance between two Gaussians."""
        cov_avg = (cov1 + cov2) / 2.0
        term1 = 0.125 * (mu1 - mu2) @ np.linalg.inv(cov_avg) @ (mu1 - mu2)
        term2 = 0.5 * np.log(
            np.linalg.det(cov_avg) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
        )
        return term1 + term2

    bhatt_matrix = np.zeros((14, 14))
    for i in range(14):
        for j in range(i + 1, 14):
            dist = bhattacharyya_distance(means[i], covs[i], means[j], covs[j])
            bhatt_matrix[i, j] = dist
            bhatt_matrix[j, i] = dist

    # generate figure
    Path(config["figures_dir"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, clf.predict(X_test))
    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix (Test Set)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    plt.colorbar(im0, ax=axes[0])

    # Bhattacharyya heatmap
    im1 = axes[1].imshow(bhatt_matrix, cmap="YlOrRd")
    axes[1].set_title("Bhattacharyya Distance Matrix")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Class")
    plt.colorbar(im1, ax=axes[1])

    fig_path = f"{config['figures_dir']}/separability.png"
    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close()

    print(
        f"separability: test_acc={test_acc:.4f}; "
        f"bhattacharyya mean={bhatt_matrix[np.triu_indices(14, k=1)].mean():.4f}"
    )

    if test_acc < 0.5:
        print(
            f"warning: test accuracy {test_acc:.4f} < 0.5; classes may not be well-separated"
        )


def mode_cond_flow(config: dict, device: str, force: bool) -> None:
    """train class-conditional velocity MLP on PCA-standardized codes.

    procedure:
        load embeddings dict and basis.
        apply basis to get codes.
        instantiate ClassCondVelocityMLP.
        delete existing checkpoint if force.
        ensure ckpt_dir exists.
        call train_class_cond_flow.
        print confirmation.
    """
    data = torch.load(
        f"{config['data_dir']}/embeddings.pt", map_location="cpu"
    , weights_only=False)
    basis = torch.load(f"{config['data_dir']}/pca_basis.pt", weights_only=False)

    codes = apply_basis(data["embeddings"], basis).float().cpu()
    labels_t = data["labels"].long().cpu()

    model = ClassCondVelocityMLP(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        hidden_dim=config["cond_flow_hidden_dim"],
    )

    ckpt_path = f"{config['ckpt_dir']}/cond_flow.pt"

    if force:
        Path(ckpt_path).unlink(missing_ok=True)

    Path(config["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    train_class_cond_flow(
        model,
        codes,
        labels_t,
        total_steps=config["cond_flow_total_steps"],
        batch_size=config["cond_flow_batch_size"],
        lr=config["cond_flow_lr"],
        device=device,
        ckpt_path=ckpt_path,
        ema_decay=config["cond_flow_ema_decay"],
    )

    print(f"trained cond_flow saved to {ckpt_path}")


def mode_log_p_y(config: dict, device: str, force: bool) -> None:
    """compute per-sample, per-class log probabilities via trained flow; cache with flow hash.

    procedure:
        load model state and basis.
        instantiate and load model.
        load or compute pstar_codes.
        compute flowhash.
        check cache; return if exists and not force.
        jacrev smoke test (4-sample validation).
        compute log_p_y [N, 14].
        atomic save.
        print confirmation.
    """
    state = torch.load(f"{config['ckpt_dir']}/cond_flow.pt", map_location="cpu", weights_only=False)
    basis = torch.load(f"{config['data_dir']}/pca_basis.pt", weights_only=False)

    model = ClassCondVelocityMLP(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        hidden_dim=config["cond_flow_hidden_dim"],
    )
    model.load_state_dict(state)
    model = model.to(device).eval()

    # load or compute pstar_codes
    pstar_cache = f"{config['data_dir']}/pstar_codes.pt"
    if Path(pstar_cache).exists() and not force:
        pstar_codes = torch.load(pstar_cache, map_location="cpu", weights_only=False)
    else:
        ds = get_dbpedia_dataset("train", cache_dir=config["hf_cache_dir"])
        balanced = np.ones(14) / 14.0
        idx = subsample_dbpedia(
            ds,
            balanced,
            K=14,
            min_per_class=config["num_samples"] // 14,
            seed=config["seed"],
        )
        texts = [ds[int(i)]["content"] for i in idx[: config["num_samples"]]]
        emb = encode_corpus(
            texts,
            model_name=config["sbert_model"],
            batch_size=config["sbert_batch_size"],
            device=device,
        )
        pstar_codes = apply_basis(emb, basis).float()

        tmp = pstar_cache + ".tmp"
        torch.save(pstar_codes.cpu(), tmp)
        os.replace(tmp, pstar_cache)

    # compute flowhash
    fh = flow_state_hash(f"{config['ckpt_dir']}/cond_flow.pt")
    log_p_y_path = f"{config['data_dir']}/log_p_y.{fh}.pt"

    if Path(log_p_y_path).exists() and not force:
        log_p_y = torch.load(log_p_y_path, weights_only=False)
        print("loaded cached log_p_y")
        return

    # jacrev smoke test (4-sample validation)
    try:
        idx_test = torch.randint(0, pstar_codes.shape[0], (4,))
        z_test = pstar_codes[idx_test].to(device)
        for i in range(4):
            y_test = torch.tensor([0], dtype=torch.long)
            log_p = log_prob_class_cond(
                model,
                z_test[i : i + 1],
                y_test.to(device),
                steps=config["log_prob_steps"],
                device=device,
                chunk_size=config["log_prob_chunk_size"],
            )
            assert log_p.shape[0] == 1
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(
            "jacrev under no_grad failed; likely PyTorch version regression. "
            "See log_prob_class_cond.py:99-102"
        ) from e

    # compute log_p_y [N, 14]
    log_p_y_list = []
    for k in range(14):
        y_k = torch.full(
            (pstar_codes.shape[0],), k, dtype=torch.long
        )
        log_p_k = log_prob_class_cond(
            model,
            pstar_codes.to(device),
            y_k.to(device),
            steps=config["log_prob_steps"],
            device=device,
            chunk_size=config["log_prob_chunk_size"],
        ).cpu()
        log_p_y_list.append(log_p_k)

    log_p_y = torch.stack(log_p_y_list, dim=1)  # [N, 14]

    tmp = log_p_y_path + ".tmp"
    torch.save(log_p_y, tmp)
    os.replace(tmp, log_p_y_path)

    print(f"computed log_p_y{tuple(log_p_y.shape)}; saved to {log_p_y_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DBpedia cond-flow pretrain: encode|pca|separability|cond_flow|log_p_y"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["encode", "pca", "separability", "cond_flow", "log_p_y"],
        help="mode to run",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--force",
        action="store_true",
        help="force recompute (ignore cache)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/dbpedia_eldr_cond_flow/config.yaml",
        help="path to config yaml",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    config = expand_paths(config)

    Path(config["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["figures_dir"]).mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    if args.mode == "encode":
        mode_encode(config, device, args.force)
    elif args.mode == "pca":
        mode_pca(config, args.force)
    elif args.mode == "separability":
        mode_separability(config, device, args.force)
    elif args.mode == "cond_flow":
        mode_cond_flow(config, device, args.force)
    elif args.mode == "log_p_y":
        mode_log_p_y(config, device, args.force)

    print("done")
