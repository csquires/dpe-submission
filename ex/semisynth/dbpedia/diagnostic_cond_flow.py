"""diagnostic plots for the dbpedia class-conditional cond_flow model.

mirrors experiments/mnist/diagnostic_cond_flow.py with two
substitutions forced by the modality:
  - sbert is one-way (no decoder), so the mnist VAE-recon and decoded-grid
    panels are replaced with code-space comparisons.
  - per-class NLL uses the dbpedia test split encoded through SBERT+PCA,
    not raw images.

produces a single figure `cond_flow_diagnostic.png` with four panels:
  A. PCA of real DBpedia codes vs cond_flow samples (overlaid, color = class)
  B. cond_flow per-class sample mean / centroid vs real per-class centroid
     (L2 distance bar chart; small bars => flow is matching the real cluster)
  C. PCA of cond_flow latent samples colored by class y (analogue of mnist C)
  D. per-class NLL on dbpedia test split (encoded via sbert + pca-standardize)

usage:
  python -m experiments.dbpedia.diagnostic_cond_flow [--config PATH]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA

from src.models.flow import (
    ClassCondVelocityMLP,
    sample_class_cond_flow,
    log_prob_class_cond,
)
from experiments.utils.dbpedia_imbalance import (
    DBPEDIA_LABEL_NAMES,
    get_dbpedia_dataset,
    subsample_dbpedia,
)
from experiments.utils.pca import apply_basis
from experiments.utils.sbert import encode_corpus


K = 14


def parse_args():
    p = argparse.ArgumentParser(description="dbpedia class-conditional flow diagnostics")
    p.add_argument("--config",
                   default="experiments/dbpedia/config.yaml",
                   help="path to config yaml")
    p.add_argument("--n-real-per-class", type=int, default=200,
                   help="real codes per class for PCA panel A and centroid panel B")
    p.add_argument("--n-pca-per-class", type=int, default=200,
                   help="cond-flow latent samples per class for PCA (panel C)")
    p.add_argument("--n-nll-per-class", type=int, default=100,
                   help="test-set samples per class for NLL (panel D)")
    p.add_argument("--sample-steps", type=int, default=100,
                   help="Euler steps for cond-flow sampling")
    p.add_argument("--log-prob-steps", type=int, default=100,
                   help="Euler steps for log_prob (smaller than production for speed)")
    return p.parse_args()


def expand_paths(config):
    """expand env var tokens in any string config value."""
    for k, v in list(config.items()):
        if isinstance(v, str) and "$" in v:
            config[k] = os.path.expandvars(v)
    return config


def load_artifacts(config, device):
    """load trained cond_flow + pca basis + cached embeddings dict.

    returns:
        flow: ClassCondVelocityMLP on device, eval mode.
        basis: dict from pca_basis.pt (numpy float32 values).
        emb_data: dict {'embeddings': [N,768] float32, 'labels': [N] int64}
                  loaded onto cpu for indexing convenience.
    """
    ckpt_dir = config["ckpt_dir"]
    data_dir = config["data_dir"]

    flow = ClassCondVelocityMLP(
        latent_dim=config["latent_dim"],
        num_classes=K,
        hidden_dim=config["cond_flow_hidden_dim"],
    )
    flow.load_state_dict(torch.load(f"{ckpt_dir}/cond_flow.pt",
                                    map_location="cpu", weights_only=False))
    flow.to(device).eval()

    basis = torch.load(f"{data_dir}/pca_basis.pt", map_location="cpu", weights_only=False)
    emb_data = torch.load(f"{data_dir}/embeddings.pt", map_location="cpu", weights_only=False)
    return flow, basis, emb_data


def real_codes_per_class(emb_data, basis, n_per_class):
    """sample n_per_class real (standardized) codes for each of K classes.

    uses the cached embeddings + basis (no SBERT call needed).

    args:
        emb_data: dict {'embeddings': [N,768] tensor, 'labels': [N] tensor}.
        basis: pca basis dict.
        n_per_class: int.

    returns:
        codes: [K * n_per_class, latent_dim] cpu float32 tensor.
        labels: [K * n_per_class] cpu int64 tensor.
    """
    emb = emb_data["embeddings"]
    labels = emb_data["labels"].long()
    rng = np.random.default_rng(0)
    pieces, lbls = [], []
    for k in range(K):
        idx_k = (labels == k).nonzero(as_tuple=True)[0].numpy()
        pick = rng.choice(idx_k, size=min(n_per_class, len(idx_k)), replace=False)
        pieces.append(emb[pick])
        lbls.append(np.full(len(pick), k, dtype=np.int64))
    raw = torch.cat(pieces, dim=0)
    codes = apply_basis(raw, basis).float()
    return codes, torch.from_numpy(np.concatenate(lbls))


def sample_cond_flow_per_class(flow, n_per_class, latent_dim, device, steps):
    """draw n_per_class flow samples per class. returns [K*n, D] cpu, labels [K*n]."""
    pieces, lbls = [], []
    with torch.no_grad():
        for y in range(K):
            z = sample_class_cond_flow(flow, y, n_per_class, latent_dim,
                                       device=device, steps=steps).cpu()
            pieces.append(z)
            lbls.append(np.full(n_per_class, y, dtype=np.int64))
    return torch.cat(pieces, dim=0), torch.from_numpy(np.concatenate(lbls))


def panel_pca_overlay(ax, real_codes, real_labels, flow_codes, flow_labels):
    """panel A: PCA fit on real codes; scatter real (circles) vs flow (x's)
    with class color (tab14-ish via tab20)."""
    pca = PCA(n_components=2).fit(real_codes.numpy())
    real_2d = pca.transform(real_codes.numpy())
    flow_2d = pca.transform(flow_codes.numpy())
    cmap = plt.get_cmap("tab20", K)
    for k in range(K):
        rmask = real_labels.numpy() == k
        fmask = flow_labels.numpy() == k
        ax.scatter(real_2d[rmask, 0], real_2d[rmask, 1],
                   s=8, color=cmap(k), alpha=0.4, marker="o")
        ax.scatter(flow_2d[fmask, 0], flow_2d[fmask, 1],
                   s=8, color=cmap(k), alpha=0.6, marker="x")
    ax.set_title(f"A. PCA: real (o) vs cond_flow (x), n={len(real_codes)//K}/class")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")


def panel_centroid_distance(ax, real_codes, real_labels,
                            flow_codes, flow_labels):
    """panel B: per-class L2 distance between real centroid and flow centroid.

    standardized space, so distances are dimensionless. small bars => flow's
    class-conditional mean lines up with the real-data class mean.
    """
    dists = np.zeros(K)
    for k in range(K):
        rmean = real_codes[real_labels == k].mean(dim=0).numpy()
        fmean = flow_codes[flow_labels == k].mean(dim=0).numpy()
        dists[k] = np.linalg.norm(rmean - fmean)
    bars = ax.bar(np.arange(K), dists, color="steelblue")
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([n[:6] for n in DBPEDIA_LABEL_NAMES],
                       rotation=60, fontsize=7, ha="right")
    ax.set_ylabel("|| real_mean - flow_mean ||_2")
    ax.set_title("B. per-class centroid distance (standardized space)")
    ax.axhline(np.mean(dists), color="black", linestyle="--", linewidth=0.8,
               label=f"mean={np.mean(dists):.3f}")
    ax.legend(fontsize=8, loc="best")


def panel_pca_by_class(ax, flow_codes, flow_labels):
    """panel C: PCA of flow samples colored by class (mnist's panel C)."""
    pca = PCA(n_components=2).fit_transform(flow_codes.numpy())
    cmap = plt.get_cmap("tab20", K)
    for k in range(K):
        mask = flow_labels.numpy() == k
        ax.scatter(pca[mask, 0], pca[mask, 1], s=6, color=cmap(k),
                   label=DBPEDIA_LABEL_NAMES[k][:8], alpha=0.6)
    ax.set_title(f"C. PCA of p_phi(z|y) samples (n={(flow_labels == 0).sum().item()}/class)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=6, ncol=2, loc="best", markerscale=1.5)


def panel_per_class_nll(ax, flow, basis, config, device,
                        n_per_class, log_prob_steps):
    """panel D: mean NLL per class on dbpedia test split.

    procedure:
        1. load dbpedia test split via wrapper.
        2. for each class k: pick first n_per_class indices.
        3. encode via sbert, project + standardize via pca basis.
        4. evaluate log_prob_class_cond at y=k.
        5. plot bar chart of mean NLL per class.
    """
    ds = get_dbpedia_dataset(split="test", cache_dir=config["hf_cache_dir"])
    labels_arr = np.asarray(ds["label"])
    nll_per_class = np.full(K, np.nan, dtype=np.float32)
    for k in range(K):
        idx_k = np.where(labels_arr == k)[0][:n_per_class]
        if len(idx_k) == 0:
            continue
        texts = [ds[int(i)]["content"] for i in idx_k]
        emb = encode_corpus(
            texts,
            model_name=config["sbert_model"],
            batch_size=min(config["sbert_batch_size"], 64),
            device=device,
        )  # [m, 768] cpu float32
        codes = apply_basis(emb, basis).to(device).float()  # [m, D]
        y_t = torch.full((len(idx_k),), k, dtype=torch.long, device=device)
        lp = log_prob_class_cond(
            flow, codes, y_t,
            steps=log_prob_steps, device=device,
            chunk_size=config.get("log_prob_chunk_size", 250),
        )
        nll_per_class[k] = -lp.mean().item()
    ax.bar(np.arange(K), nll_per_class, color="steelblue")
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([n[:6] for n in DBPEDIA_LABEL_NAMES],
                       rotation=60, fontsize=7, ha="right")
    ax.set_ylabel("mean NLL (nats)")
    mean_nll = np.nanmean(nll_per_class)
    ax.axhline(mean_nll, color="black", linestyle="--", linewidth=0.8,
               label=f"mean={mean_nll:.2f}")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(f"D. test-set NLL per class (n={n_per_class}/class)")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    config = expand_paths(config)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: cuda unavailable, falling back to cpu")
        device = "cpu"
    torch.manual_seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))

    flow, basis, emb_data = load_artifacts(config, device)

    real_codes, real_labels = real_codes_per_class(
        emb_data, basis, args.n_real_per_class)
    flow_codes, flow_labels = sample_cond_flow_per_class(
        flow, args.n_real_per_class,  # match real for like-for-like overlay
        config["latent_dim"], device, args.sample_steps)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    panel_pca_overlay(axes[0, 0], real_codes, real_labels,
                      flow_codes, flow_labels)
    panel_centroid_distance(axes[0, 1], real_codes, real_labels,
                            flow_codes, flow_labels)
    # panel C uses denser per-class samples for a cleaner picture
    flow_codes_dense, flow_labels_dense = sample_cond_flow_per_class(
        flow, args.n_pca_per_class,
        config["latent_dim"], device, args.sample_steps)
    panel_pca_by_class(axes[1, 0], flow_codes_dense, flow_labels_dense)
    panel_per_class_nll(axes[1, 1], flow, basis, config, device,
                        args.n_nll_per_class, args.log_prob_steps)

    fig.tight_layout()
    fig_dir = Path(config["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "cond_flow_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
