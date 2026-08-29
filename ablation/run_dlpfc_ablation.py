#!/usr/bin/env python
# Auto-detect repo root for reproducibility
import os as _os, sys as _sys
_CURRENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_CURRENT_DIR)
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = _os.path.join(REPO_ROOT, 'saved_results')
_sys.path.insert(0, _os.path.join(REPO_ROOT, 'dcpbst_package'))
_ABLATION_DIR = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ABLATION_DIR)
import argparse
import gc
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import homogeneity_completeness_v_measure
# Paths derived from REPO_ROOT (set in setup block)
DATA_ROOT = os.path.join(DATA_DIR)
IMAGE_CACHE_DIR = os.path.join(REPO_ROOT, "data", "cut_img_DLPFC")
ABLATION_OUTPUT_DIR = os.path.join(REPO_ROOT, "saved_results", "DLPFC_ablation")
RESULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "saved_results", "DLPFC_data")
# sys.path configured in setup block
from dcpbst_package.model_827_copy import Dcpbst as FullModel
# from dcpbst_package.model_katz_quchu import Miso as SpatialTopologyAblationMiso  # Ablation variant file not included in this package
# from dcpbst_package.model_qurongyu_xiaorong import Miso as RedundancyAblationMiso  # Ablation variant file not included in this package
# from dcpbst_package.model_qurongyu_zhuyili import Miso as CrossAttentionAblationMiso  # Ablation variant file not included in this package
from dcpbst_package.utils import clustering as miso_clustering
from dcpbst_package.model import Dcpbst
from dcpbst_package.utils import clustering as dcpbst_clustering
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# NOTE: This ablation study originally used 4 model variants:
#   1. FullModel (Dcpbst) — included in dcpbst_package/model_827_copy.py ✓
#   2. SpatialTopologyAblationMiso — NOT included (requires model_katz_quchu.py)
#   3. RedundancyAblationMiso — NOT included (requires model_qurongyu_xiaorong.py)
#   4. CrossAttentionAblationMiso — NOT included (requires model_qurongyu_zhuyili.py)
# Only the FullModel (complete model) is available in this reproducibility package.
# To run the full ablation, the original ablation model variant files are required.
@dataclass
class MethodConfig:
    key: str
    display_name: str
    csv_name: str
    model_cls: type
    cluster_fn: callable
    model_kwargs: Dict = field(default_factory=dict)
    fit_overrides: Dict = field(default_factory=dict)
    duplicate_to_data_dir: bool = False
    data_dir_csv_name: Optional[str] = None
SLICE_IDS = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]
DEFAULT_FIT_PARAMS = {
    "epochs": 900,
    "lr": 1e-3,
    "w_cls": 10.0,
    "w_recon": 10.0,
    "w_kl": 0.1,
    "w_pro": 3.0,
    "w_info": 2.0,
    "w_dgi": 0.1,
    "w_clu": 1.0,
    "diagnose_every_n_epochs": 20,
}
METHOD_ORDER = [
    "DCPBST",
    "w/o Cross_Attention",
    "w/o Imbalance_Regulation",
    "w/o L_DGI",
    "w/o L_InfoNCE",
    "w/o Redundancy_Reduction",
    "w/o Spatial_Topology",
]
METHOD_CONFIGS = {
    "dcpbst": MethodConfig(
        key="dcpbst",
        display_name="DCPBST",
        csv_name="DLPFC_batch_DCRBST_results.csv",
        model_cls=Dcpbst,
        cluster_fn=dcpbst_clustering,
        duplicate_to_data_dir=True,
        data_dir_csv_name="DLPFC_batch_DCRBST_results_revalidated.csv",
    ),
    # "cross_attention": MethodConfig(
    #     key="cross_attention",
    #     display_name="w/o Cross_Attention",
    #     csv_name="DLPFC_batch_Miso_results_zhuyilixiaorong.csv",
    #     model_cls=CrossAttentionAblationMiso,  # Ablation variant file not included
    #     cluster_fn=miso_clustering,
    #     model_kwargs={"use_cross_attention": False},
    # ),
    "imbalance_regulation": MethodConfig(
        key="imbalance_regulation",
        display_name="w/o Imbalance_Regulation",
        csv_name="DLPFC_batch_Miso_results_tiaoquanxiaorong.csv",
        model_cls=FullModel,
        cluster_fn=miso_clustering,
        model_kwargs={"use_imbalance_regulation": False},
    ),
    "l_dgi": MethodConfig(
        key="l_dgi",
        display_name="w/o L_DGI",
        csv_name="DLPFC_batch_Miso_results_dgixiaorong.csv",
        model_cls=FullModel,
        cluster_fn=miso_clustering,
        fit_overrides={"w_dgi": 0.0},
    ),
    "l_infonce": MethodConfig(
        key="l_infonce",
        display_name="w/o L_InfoNCE",
        csv_name="DLPFC_batch_Miso_results_INFOxiaorong.csv",
        model_cls=FullModel,
        cluster_fn=miso_clustering,
        fit_overrides={"w_info": 0.0},
    ),
    # "redundancy_reduction": MethodConfig(
    #     key="redundancy_reduction",
    #     display_name="w/o Redundancy_Reduction",
    #     csv_name="DLPFC_batch_Miso_results_qurongyuxiaorong.csv",
    #     model_cls=RedundancyAblationMiso,  # Ablation variant file not included
    #     cluster_fn=miso_clustering,
    #     model_kwargs={"use_redundancy_removal": False},
    # ),
    # "spatial_topology": MethodConfig(
    #     key="spatial_topology",
    #     display_name="w/o Spatial_Topology",
    #     csv_name="DLPFC_batch_Miso_results_katzxiaorong.csv",
    #     model_cls=SpatialTopologyAblationMiso,  # Ablation variant file not included
    #     cluster_fn=miso_clustering,
    # ),
}
def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def load_visium_with_labels(path: str, label_col: str = "layer_guess"):
    adata = sc.read_visium(path, load_images=True)
    adata.var_names_make_unique()
    metadata_path = os.path.join(path, "metadata.tsv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.tsv not found in {path}")
    meta_df = pd.read_csv(metadata_path, sep="\t")
    if label_col not in meta_df.columns:
        raise KeyError(f"{label_col} not in metadata.tsv for {path}")
    adata.obs["Ground Truth"] = meta_df[label_col].values
    adata = adata[~pd.isnull(adata.obs["Ground Truth"])].copy()
    return adata
def preprocess_rna_dlpfc(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata[:, adata.var["highly_variable"]].copy()
def load_cached_image_features(section_id: str, cache_dir: str = IMAGE_CACHE_DIR) -> np.ndarray:
    feature_path = os.path.join(cache_dir, f"Img_feat_{section_id}.npy")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Missing cached image features for {section_id}: {feature_path}"
        )
    return np.load(feature_path)
def get_scrna_matrix(adata) -> np.ndarray:
    if scipy.sparse.issparse(adata.X):
        return adata.X.A
    return np.asarray(adata.X)
def purity_score(y_true, y_pred) -> float:
    cm = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))
def build_result_df(sample: str, methods_name: str, y_pred, ground_truth) -> pd.DataFrame:
    ari = adjusted_rand_score(y_pred, ground_truth)
    nmi = normalized_mutual_info_score(y_pred, ground_truth)
    purity = purity_score(y_pred, ground_truth)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        y_pred, ground_truth
    )
    print(
        f"ARI={ari}, NMI={nmi}, Purity={purity}, V_Measure={v_measure}"
    )
    return pd.DataFrame(
        [
            {
                "Sample": sample,
                "ARI": ari,
                "NMI": nmi,
                "Purity": purity,
                "Homogeneity": homogeneity,
                "Completeness": completeness,
                "V_Measure": v_measure,
                "methods": methods_name,
            }
        ]
    )
def build_model(method: MethodConfig, scrna: np.ndarray, image_emb: np.ndarray, adata, device: str):
    n_clusters = len(set(adata.obs["Ground Truth"]))
    model = method.model_cls(
        [scrna, image_emb],
        sparse=False,
        device=device,
        n_clusters=n_clusters,
        adata=adata,
        neighbors=7,
        **method.model_kwargs,
    )
    return model, n_clusters
def evaluate_single_slice(
    section_id: str,
    method: MethodConfig,
    fit_params: Dict,
    device: str,
    label_col: str = "layer_guess",
) -> pd.DataFrame:
    data_path = os.path.join(DATA_ROOT, section_id)
    adata = load_visium_with_labels(data_path, label_col=label_col)
    adata = preprocess_rna_dlpfc(adata)
    scrna = get_scrna_matrix(adata)
    adata.obsm["feat_img"] = load_cached_image_features(section_id)
    model, n_clusters = build_model(method, scrna, adata.obsm["feat_img"], adata, device)
    merged_fit_params = dict(fit_params)
    merged_fit_params.update(method.fit_overrides)
    print(f"[{method.display_name}] Running section {section_id} with params: {merged_fit_params}")
    embedding = model.fit(**merged_fit_params)
    adata.obsm["emb"] = embedding
    method.cluster_fn(
        adata,
        n_clusters=n_clusters,
        radius=50,
        method="mclust",
        refinement=True,
    )
    result_df = build_result_df(
        sample=section_id,
        methods_name=method.display_name,
        y_pred=adata.obs["domain"],
        ground_truth=adata.obs["Ground Truth"],
    )
    del model
    del adata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result_df
def save_method_results(method: MethodConfig, result_df: pd.DataFrame, ablation_dir: str, data_dir: str) -> None:
    os.makedirs(ablation_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    ablation_csv_path = os.path.join(ablation_dir, method.csv_name)
    result_df.to_csv(ablation_csv_path, index=False)
    print(f"Saved {method.display_name} results to {ablation_csv_path}")
    if method.duplicate_to_data_dir and method.data_dir_csv_name:
        data_csv_path = os.path.join(data_dir, method.data_dir_csv_name)
        result_df.to_csv(data_csv_path, index=False)
        print(f"Saved {method.display_name} verification results to {data_csv_path}")
def create_boxplot_figure(csv_dir: str, output_path: str) -> None:
    csv_files = sorted(
        [
            os.path.join(csv_dir, name)
            for name in os.listdir(csv_dir)
            if name.endswith(".csv")
        ]
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    combined_df = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
    combined_df = combined_df[combined_df["methods"].isin(METHOD_ORDER)].copy()
    combined_df["methods"] = pd.Categorical(
        combined_df["methods"], categories=METHOD_ORDER, ordered=True
    )
    rcParams["font.family"] = "Arial"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "SimHei"]
    rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    metrics = ["ARI", "NMI", "Purity", "V_Measure"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()
    for index, metric in enumerate(metrics):
        ax = axes[index]
        sns.boxplot(
            y="methods",
            x=metric,
            data=combined_df,
            ax=ax,
            order=METHOD_ORDER,
            orient="h",
        )
        ax.set_title(metric, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
    fig.subplots_adjust(
        left=0.22,
        right=0.98,
        bottom=0.08,
        top=0.94,
        wspace=0.32,
        hspace=0.28,
    )
    fig.align_ylabels(axes)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved boxplot figure to {output_path}")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DLPFC ablation2 experiments.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_CONFIGS.keys()),
        choices=list(METHOD_CONFIGS.keys()),
        help="Method keys to run.",
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        default=SLICE_IDS,
        help="DLPFC slice ids to process.",
    )
    parser.add_argument("--seed", type=int, default=100, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Training device.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_FIT_PARAMS["epochs"],
        help="Training epochs override.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip final boxplot generation.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    fit_params = dict(DEFAULT_FIT_PARAMS)
    fit_params["epochs"] = args.epochs
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in the current environment.")
    selected_methods = [METHOD_CONFIGS[key] for key in args.methods]
    print(f"Running methods: {[method.display_name for method in selected_methods]}")
    print(f"Running slices: {args.slices}")
    print(f"Using device: {args.device}")
    for method in selected_methods:
        method_results: List[pd.DataFrame] = []
        for section_id in args.slices:
            print(f"\n{'=' * 30} {method.display_name} | {section_id} {'=' * 30}")
            try:
                result_df = evaluate_single_slice(
                    section_id=section_id,
                    method=method,
                    fit_params=fit_params,
                    device=args.device,
                )
                method_results.append(result_df)
            except Exception as exc:
                print(f"Failed on {method.display_name} / {section_id}: {exc}")
                continue
        if not method_results:
            print(f"No successful runs for {method.display_name}")
            continue
        final_df = pd.concat(method_results, ignore_index=True)
        final_df = final_df[
            [
                "Sample",
                "methods",
                "ARI",
                "NMI",
                "Purity",
                "Homogeneity",
                "Completeness",
                "V_Measure",
            ]
        ]
        save_method_results(method, final_df, ABLATION_OUTPUT_DIR, RESULT_OUTPUT_DIR)
        metric_means = final_df[["ARI", "NMI", "Purity", "V_Measure"]].mean().round(4)
        print(f"Mean metrics for {method.display_name}:\n{metric_means}")
    if not args.skip_plot:
        plot_path = os.path.join(ABLATION_OUTPUT_DIR, "DLPFC_ablation2_boxplots.png")
        create_boxplot_figure(ABLATION_OUTPUT_DIR, plot_path)
if __name__ == "__main__":
    main()