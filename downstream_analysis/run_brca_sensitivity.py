#!/usr/bin/env python
"""
BRCA dataset parameter sensitivity ablation experiment.

Tests three hyperparameters on BRCA (Human_breast):
  - w_kl                     in [0.1, 0.3, 0.5, 0.7, 1.0]   (5 values)
  - w_dgi                    in [0.1, 0.3, 0.5, 0.7, 1.0]   (5 values)
  - diagnose_every_n_epochs  in [20, 30, 40, 50, 60, 70]    (6 values)

Univariate ablation: change one parameter at a time, keep others at default.
Total runs: 5 + 5 + 6 = 16.

Usage:
    python run_brca_sensitivity.py --device cuda
    python run_brca_sensitivity.py --data_path data/Human_breast --device cpu
"""
# Auto-detect repo root for reproducibility
import os as _os, sys as _sys
_CURRENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_CURRENT_DIR)
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = _os.path.join(REPO_ROOT, 'saved_results', 'sensitivity')
_sys.path.insert(0, _os.path.join(REPO_ROOT, 'dcpbst_package'))

import argparse
import random
import traceback
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score,
                             homogeneity_completeness_v_measure,
                             normalized_mutual_info_score)
from sklearn.metrics.cluster import contingency_matrix
from tqdm import tqdm

import ot

# DCPBST package
from dcpbst_package.model_827_copy import Dcpbst
from dcpbst_package.hist_features import get_features

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 100
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
Image.MAX_IMAGE_PIXELS = None


def build_device():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("CUDA not available. Using CPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_visium_with_labels(path: str, label_col: str = "fine_annot_type"):
    """Load Visium and attach ground-truth labels from metadata.tsv."""
    adata = sc.read_visium(path, load_images=True)
    adata.var_names_make_unique()
    meta_path = _os.path.join(path, "metadata.tsv")
    if not _os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.tsv not found in {path}")
    meta = pd.read_csv(meta_path, sep="\t")
    assert label_col in meta.columns, f"{label_col} not in metadata.tsv"
    adata.obs["Ground Truth"] = meta[label_col].values
    adata = adata[~pd.isnull(adata.obs["Ground Truth"])]
    return adata


def preprocess_rna(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata[:, adata.var["highly_variable"]]


def extract_image_features_resnet(
    adata, data_path: str, section_id: str = "BRCA", backbone: str = "ResNet50",
    device: str = "cpu"
):
    """Extract H&E image patches + ResNet50 features (cached on disk)."""
    import torchvision.models as models

    if backbone == "ResNet50":
        img_model = models.resnet50(pretrained=True).to(device)

    full_image = cv2.imread(_os.path.join(data_path, "spatial", "full_image.tif"))
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

    save_dir = _os.path.join(DATA_DIR, "cut_img_BRCA")
    _os.makedirs(save_dir, exist_ok=True)
    cut_img_path = _os.path.join(save_dir, "cut_img_BRCA.npy")
    img_feat_path = _os.path.join(save_dir, "Img_feat_brca.npy")

    img_size = 224
    if not _os.path.exists(cut_img_path):
        patches = []
        for x, y in adata.obsm["spatial"]:
            patch = full_image[y - img_size:y + img_size, x - img_size:x + img_size]
            patches.append(patch)
        np.save(cut_img_path, np.array(patches))
        print("Cutting image finished!")

    if not _os.path.exists(img_feat_path):
        spot_img = np.load(cut_img_path).astype(np.float32) / 255.0
        tensor = torch.from_numpy(spot_img)
        img_feat = []
        for element in tqdm(tensor, desc=f"Extracting features for {section_id}"):
            element = element.resize_(1, 3, 224, 224).to(device)
            ret = img_model(element)
            img_feat.append(ret.data.cpu().numpy().ravel())
        np.save(img_feat_path, np.array(img_feat))
    else:
        img_feat = np.load(img_feat_path)

    print("Image feature extraction complete!")
    adata.obsm["feat_img"] = img_feat
    return adata


# ---------------------------------------------------------------------------
# Metrics & clustering
# ---------------------------------------------------------------------------
def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def calculate_clustering_matrix(y_pred, ground_truth):
    ari = adjusted_rand_score(ground_truth, y_pred)
    nmi = normalized_mutual_info_score(ground_truth, y_pred)
    purity = purity_score(ground_truth, y_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        ground_truth, y_pred
    )
    print(f"ARI={ari:.4f}, NMI={nmi:.4f}, Purity={purity:.4f}, V-Measure={v_measure:.4f}")
    return pd.DataFrame([{
        "ARI": ari, "NMI": nmi, "Purity": purity,
        "Homogeneity": homogeneity, "Completeness": completeness,
        "V_Measure": v_measure,
    }])


def search_res(adata, n_clusters, method="leiden", use_rep="emb",
               start=0.1, end=3.0, increment=0.01):
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in np.arange(start, end, increment):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count = len(adata.obs["leiden"].unique())
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count = len(adata.obs["louvain"].unique())
        else:
            continue
        if count == n_clusters:
            print(f"Found resolution: {res} for {n_clusters} clusters.")
            return res
    raise ValueError("Resolution not found in the given range.")


def refine_label(adata, radius=50, key="label"):
    """Spatial neighbour majority-vote label smoothing."""
    position = adata.obsm["spatial"]
    distance = ot.dist(position, position, metric="euclidean")
    old_type = adata.obs[key].values
    new_type = []
    for i in range(distance.shape[0]):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = old_type[index[1:radius + 1]]
        max_type = max(list(neigh_type), key=list(neigh_type).count)
        new_type.append(max_type)
    return [str(i) for i in new_type]



def mclust_R(adata, num_cluster, used_obsm="emb_pca", random_seed=2024):
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r.library("mclust")
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster)
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = pd.Categorical(mclust_res.astype('int'))
    return adata


def clustering(adata, n_clusters=7, radius=50, key="emb",
               method="mclust", refinement=False):
    """PCA on the learned embedding, then mclust/KMeans clustering and
    optional spatial label smoothing. Writes adata.obs['domain']."""
    pca = PCA(n_components=20, random_state=2024)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm["emb_pca"] = embedding
    if method == "mclust":
        adata = mclust_R(adata, num_cluster=n_clusters, used_obsm="emb_pca")
        adata.obs["domain"] = adata.obs["mclust"]
    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=2024).fit(embedding)
        adata.obs["domain"] = [str(i + 1) for i in kmeans.labels_]
    if refinement:
        adata.obs["domain"] = refine_label(adata, radius, key="domain")
    return adata


def run_dcpbst_pipeline(data_path, device="cuda", label_col="fine_annot_type",
                        fit_params=None):
    """Run the full DCPBST training + clustering pipeline on BRCA."""
    adata = load_visium_with_labels(data_path, label_col)
    print(f"Loaded data for BRCA: {adata.shape}")
    adata = preprocess_rna(adata)

    scrna = adata.X.A if hasattr(adata.X, "A") else adata.X

    adata = extract_image_features_resnet(
        adata, data_path=data_path, section_id="BRCA", device=device
    )
    image_emb = adata.obsm["feat_img"]

    n_clusters = len(adata.obs["Ground Truth"].unique())
    print(f"Number of clusters: {n_clusters}")

    model = Dcpbst(
        [scrna, image_emb], sparse=False, device=device,
        n_clusters=n_clusters, adata=adata, neighbors=7,
    )

    final_params = {**DEFAULT_FIT_PARAMS, **(fit_params or {})}
    embedding = model.fit(**final_params)
    adata.obsm["emb"] = embedding

    adata = clustering(
        adata, n_clusters=n_clusters, method="mclust", refinement=True
    )
    eval_df = calculate_clustering_matrix(
        adata.obs["domain"], adata.obs["Ground Truth"]
    )
    return adata, model, eval_df


# ---------------------------------------------------------------------------
# Sensitivity experiments
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "w_kl":                    [0.1, 0.3, 0.5, 0.7, 1.0],
    "w_dgi":                   [0.1, 0.3, 0.5, 0.7, 1.0],
    "diagnose_every_n_epochs": [20, 30, 40, 50, 60, 70],
}

DEFAULT_VALUES = {
    "w_kl": 0.1,
    "w_dgi": 0.1,
    "diagnose_every_n_epochs": 30,
}


def run_parameter_sensitivity_experiment(data_path, device="cuda"):
    """Univariate ablation — one parameter varies at a time."""
    all_results = {k: [] for k in PARAM_GRID}

    for param_name, values in PARAM_GRID.items():
        print("\n" + "=" * 60)
        print(f"Parameter sensitivity: {param_name}")
        print("=" * 60)

        for val in values:
            print(f"\nTest {param_name} = {val}")
            print("-" * 60)

            fit_params = dict(DEFAULT_VALUES)
            fit_params[param_name] = val

            try:
                adata, model, eval_df = run_dcpbst_pipeline(
                    data_path, device=device, fit_params=fit_params,
                )
                eval_df["Sample"] = "BRCA"
                eval_df[param_name] = val

                all_results[param_name].append({
                    "parameter": param_name,
                    "parameter_value": val,
                    "ARI": eval_df["ARI"].iloc[0],
                    "NMI": eval_df["NMI"].iloc[0],
                    "Purity": eval_df["Purity"].iloc[0],
                    "Homogeneity": eval_df["Homogeneity"].iloc[0],
                    "Completeness": eval_df["Completeness"].iloc[0],
                    "V_Measure": eval_df["V_Measure"].iloc[0],
                })
                print(f"  ARI={eval_df['ARI'].iloc[0]:.4f}  NMI={eval_df['NMI'].iloc[0]:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_parameter_sensitivity_results(experiment_results,
                                       save_path="BRCA_sensitivity_barplot.png"):
    """Grouped bar chart: ARI + NMI for each parameter grid."""
    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "BRCA Dataset — Parameter Sensitivity Analysis (ARI & NMI)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    color_ari, color_nmi = "#FF6B6B", "#4ECDC4"

    for idx, (param_name, ax) in enumerate(zip(PARAM_GRID.keys(), axes)):
        results = experiment_results.get(param_name)
        if not results:
            continue
        df = pd.DataFrame(results).sort_values("parameter_value")
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width / 2, df["ARI"], width,
               label="ARI", color=color_ari, alpha=0.8)
        ax.bar(x + width / 2, df["NMI"], width,
               label="NMI", color=color_nmi, alpha=0.8)
        ax.set_xlabel(param_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(f"({chr(ord('a') + idx)}) {param_name} Sensitivity",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        fmt = "{:.1f}" if param_name != "diagnose_every_n_epochs" else "{:.0f}"
        ax.set_xticklabels([fmt.format(v) for v in df["parameter_value"]],
                           fontsize=10)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.set_ylim([0, 1.0])
        for i, (ari, nmi) in enumerate(zip(df["ARI"], df["NMI"])):
            ax.text(i - width / 2, ari + 0.02, f"{ari:.3f}",
                    ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, nmi + 0.02, f"{nmi:.3f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_dir = _os.path.join(REPO_ROOT, "figures")
    _os.makedirs(out_dir, exist_ok=True)
    full_path = _os.path.join(out_dir, save_path)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {full_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path",
                        default=_os.path.join(DATA_DIR, "Human_breast"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    device = args.device or build_device()
    data_path = args.data_path
    out_dir = args.output_dir

    print("=" * 60)
    print("BRCA parameter sensitivity ablation")
    print("=" * 60)
    print(f"  w_kl values:                   {PARAM_GRID['w_kl']}")
    print(f"  w_dgi values:                  {PARAM_GRID['w_dgi']}")
    print(f"  diagnose_every_n_epochs values:{PARAM_GRID['diagnose_every_n_epochs']}")
    print(f"  Total runs: {sum(len(v) for v in PARAM_GRID.values())}")
    print(f"  Data:   {data_path}")
    print(f"  Output: {out_dir}")
    print(f"  Device: {device}")
    print("=" * 60)

    results = run_parameter_sensitivity_experiment(data_path, device=device)

    _os.makedirs(out_dir, exist_ok=True)
    for param_name, result_list in results.items():
        if not result_list:
            continue
        df = pd.DataFrame(result_list)
        csv_path = _os.path.join(out_dir, f"sensitivity_{param_name}_BRCA.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    summary = []
    for rlist in results.values():
        summary.extend(rlist)
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(_os.path.join(out_dir, "sensitivity_summary_BRCA.csv"),
                          index=False)
        print(f"Saved: {_os.path.join(out_dir, 'sensitivity_summary_BRCA.csv')}")

    plot_parameter_sensitivity_results(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
