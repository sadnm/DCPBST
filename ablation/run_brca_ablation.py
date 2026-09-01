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
import json
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
import torch
import torchvision.models as models
import cv2
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import homogeneity_completeness_v_measure
# Paths derived from REPO_ROOT (set in setup block)
DATA_ROOT = os.path.join(DATA_DIR)
IMAGE_CACHE_DIR = os.path.join(REPO_ROOT, "data", "cut_img_BRCA")
BRCA_DATA_PATH = os.path.join(DATA_ROOT, 'Human_breast')
ABLATION_OUTPUT_DIR = os.path.join(REPO_ROOT, 'saved_results', 'BRCA_ablation')
from dcpbst_package.model_827_copy import Dcpbst as FullModel
# from dcpbst_package.model_katz_quchu import Miso as SpatialTopologyAblationMiso  # Ablation variant file not included in this package
# from dcpbst_package.model_qurongyu_xiaorong import Miso as RedundancyAblationMiso  # Ablation variant file not included in this package
# from dcpbst_package.model_qurongyu_zhuyili import Miso as CrossAttentionAblationMiso  # Ablation variant file not included in this package
from dcpbst_package.hist_features import get_features
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
    model_kwargs: Dict = field(default_factory=dict)
    fit_overrides: Dict = field(default_factory=dict)
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
    "Full (DCPBST/Miso)",
    "w/o Spatial_Topology",
    "w/o Redundancy_Reduction",
    "w/o Cross_Attention",
    "w/o Imbalance_Regulation",
    "w/o L_InfoNCE",
    "w/o L_DGI",
]
METHOD_CONFIGS = {
    "full": MethodConfig(
        key="full",
        display_name="Full (DCPBST/Miso)",
        csv_name="BRCA_batch_Miso_full_results.csv",
        model_cls=FullModel,
    ),
    # "spatial_topology": MethodConfig(
    #     key="spatial_topology",
    #     display_name="w/o Spatial_Topology",
    #     csv_name="BRCA_batch_Miso_results_katzxiaorong.csv",
    #     model_cls=SpatialTopologyAblationMiso,  # Ablation variant file not included
    # ),
    # "redundancy_reduction": MethodConfig(
    #     key="redundancy_reduction",
    #     display_name="w/o Redundancy_Reduction",
    #     csv_name="BRCA_batch_Miso_results_qurongyuxiaorong.csv",
    #     model_cls=RedundancyAblationMiso,  # Ablation variant file not included
    #     model_kwargs={"use_redundancy_removal": False},
    # ),
    # "cross_attention": MethodConfig(
    #     key="cross_attention",
    #     display_name="w/o Cross_Attention",
    #     csv_name="BRCA_batch_Miso_results_zhuyilixiaorong.csv",
    #     model_cls=CrossAttentionAblationMiso,  # Ablation variant file not included
    #     model_kwargs={"use_cross_attention": False},
    # ),
    "imbalance_regulation": MethodConfig(
        key="imbalance_regulation",
        display_name="w/o Imbalance_Regulation",
        csv_name="BRCA_batch_Miso_results_tiaoquanxiaorong.csv",
        model_cls=FullModel,
        model_kwargs={"use_imbalance_regulation": False},
    ),
    "l_infonce": MethodConfig(
        key="l_infonce",
        display_name="w/o L_InfoNCE",
        csv_name="BRCA_batch_Miso_results_INFOxiaorong.csv",
        model_cls=FullModel,
        fit_overrides={"w_info": 0.0},
    ),
    "l_dgi": MethodConfig(
        key="l_dgi",
        display_name="w/o L_DGI",
        csv_name="BRCA_batch_Miso_results_dgixiaorong.csv",
        model_cls=FullModel,
        fit_overrides={"w_dgi": 0.0},
    ),
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
def load_visium_with_labels(path: str, label_col: str = "fine_annot_type"):
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
def extract_scalefactors(path):
    scale_path = os.path.join(path, "spatial", "scalefactors_json.json")
    with open(scale_path, "r") as f:
        scale_info = json.load(f)
    spot_diameter = scale_info["spot_diameter_fullres"]
    hires_scalef = scale_info["tissue_hires_scalef"]
    return spot_diameter, hires_scalef
def preprocess_rna_brca(adata):
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata[:, adata.var["highly_variable"]].copy()
def extract_image_features_resnet(adata, data_path, backbone='ResNet50', device='cpu'):
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    img_model = models.resnet50(pretrained=True)
    img_model.to(device)
    img_model.eval()
    full_image = cv2.imread(os.path.join(data_path, "spatial", "full_image.tif"))
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    cut_save_path = os.path.join(IMAGE_CACHE_DIR, 'cut_img_BRCA.npy')
    img_size = 224
    if not os.path.exists(cut_save_path):
        patches = []
        for x, y in adata.obsm['spatial']:
            patch = full_image[int(y - img_size):int(y + img_size), int(x - img_size):int(x + img_size)]
            if patch.shape != (img_size * 2, img_size * 2, 3):
                patch = np.zeros((img_size * 2, img_size * 2, 3), dtype=np.uint8)
            patches.append(patch)
        patches = np.array(patches)
        np.save(cut_save_path, patches)
    print("cut_img_BRCA ready")
    feat_save_path = os.path.join(IMAGE_CACHE_DIR, 'Img_feat_brca.npy')
    if not os.path.exists(feat_save_path):
        spot_img = np.load(cut_save_path)
        spot_img = spot_img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(spot_img)
        img_feat = []
        for i, element in tqdm(enumerate(tensor), total=len(tensor), desc="Extracting img feat"):
            element = element.permute(2, 0, 1).unsqueeze(0)
            element = torch.nn.functional.interpolate(element, size=(224, 224), mode='bilinear', align_corners=False)
            element = element.to(device)
            with torch.no_grad():
                ret = img_model(element)
            ret = ret.data.cpu().numpy().ravel()
            img_feat.append(ret)
        np.save(feat_save_path, img_feat)
        img_feat = np.array(img_feat)
    else:
        img_feat = np.load(feat_save_path)
    print("Img_feat_brca ready, shape:", img_feat.shape)
    adata.obsm['feat_img'] = img_feat
    return adata
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
        f"ARI={ari:.4f}, NMI={nmi:.4f}, Purity={purity:.4f}, V_Measure={v_measure:.4f}"
    )
    return pd.DataFrame(
        [
            {
                "Sample": sample,
                "methods": methods_name,
                "ARI": ari,
                "NMI": nmi,
                "Purity": purity,
                "Homogeneity": homogeneity,
                "Completeness": completeness,
                "V_Measure": v_measure,
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

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2024):
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2]).astype('int')
    adata.obs['mclust'] = pd.Categorical(mclust_res)
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', refinement=False):
    pca = PCA(n_components=20, random_state=2024)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=2024).fit(embedding)
        kmeans_result = [i + 1 for i in kmeans.labels_]
        adata.obs['domain'] = list(map(lambda x: str(x), kmeans_result))
    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type
def evaluate_single_run(
    method: MethodConfig,
    fit_params: Dict,
    device: str,
    label_col: str = "fine_annot_type",
    ablation_dir: Optional[str] = None,
) -> pd.DataFrame:
    # ---- method-level resume: skip entire run if CSV already exists and looks complete ----
    if ablation_dir is not None:
        target_csv = os.path.join(ablation_dir, method.csv_name)
        if os.path.isfile(target_csv):
            try:
                exist = pd.read_csv(target_csv)
                required = {"ARI", "NMI", "Purity", "V_Measure"}
                if required.issubset(set(exist.columns)) and len(exist) > 0:
                    print(f"[skip] {method.display_name}: already saved at {target_csv}")
                    return exist
            except Exception as e:
                print(f"[warn] could not read existing CSV {target_csv}: {e}. Will re-run.")
    adata = load_visium_with_labels(BRCA_DATA_PATH, label_col=label_col)
    adata = preprocess_rna_brca(adata)
    scrna = get_scrna_matrix(adata)
    if 'feat_img' not in adata.obsm.keys():
        adata = extract_image_features_resnet(adata, data_path=BRCA_DATA_PATH, device=device)
    image_emb = adata.obsm['feat_img']
    model, n_clusters = build_model(method, scrna, image_emb, adata, device)
    merged_fit_params = dict(fit_params)
    merged_fit_params.update(method.fit_overrides)
    # ---- epoch-level checkpoint: per-method prefix under ablation_dir/ckpts ----
    ckpt_prefix = None
    resume_from = None
    if ablation_dir is not None:
        ckpt_dir = os.path.join(ablation_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        safe_key = method.key.replace(os.sep, "_")
        ckpt_prefix = os.path.join(ckpt_dir, f"method_{safe_key}")
        last_pt = ckpt_prefix + "_last.pt"
        if os.path.isfile(last_pt):
            resume_from = last_pt
    # default save_every to 100 if user didn't set it explicitly
    merged_fit_params.setdefault("save_every", 100)
    merged_fit_params.setdefault("ckpt_path", ckpt_prefix)
    merged_fit_params.setdefault("resume_from", resume_from)
    print(f"[{method.display_name}] Running with params: {merged_fit_params}")
    embedding = model.fit(**merged_fit_params)
    adata.obsm['emb'] = embedding
    clustering(
        adata,
        n_clusters=n_clusters,
        radius=50,
        method="mclust",
        refinement=True,
    )
    result_df = build_result_df(
        sample="Human_breast",
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
def save_method_results(method: MethodConfig, result_df: pd.DataFrame, ablation_dir: str) -> None:
    os.makedirs(ablation_dir, exist_ok=True)
    ablation_csv_path = os.path.join(ablation_dir, method.csv_name)
    result_df.to_csv(ablation_csv_path, index=False)
    print(f"Saved {method.display_name} results to {ablation_csv_path}")
def create_boxplot_figure(csv_dir: str, output_path: str) -> None:
    csv_files = sorted(
        [
            os.path.join(csv_dir, name)
            for name in os.listdir(csv_dir)
            if name.endswith(".csv")
        ]
    )
    if not csv_files:
        print(f"No CSV files found in {csv_dir}, skip plotting")
        return
    combined_df = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
    combined_df = combined_df[combined_df["methods"].isin(METHOD_ORDER)].copy()
    combined_df["methods"] = pd.Categorical(
        combined_df["methods"], categories=METHOD_ORDER, ordered=True
    )
    method_colors = {
        "Full (DCPBST/Miso)": "#1f77b4",
        "w/o Spatial_Topology": "#2ca02c",
        "w/o Redundancy_Reduction": "#ff7f0e",
        "w/o Cross_Attention": "#d62728",
        "w/o Imbalance_Regulation": "#9467bd",
        "w/o L_InfoNCE": "#8c564b",
        "w/o L_DGI": "#e377c2",
    }
    rcParams["font.family"] = "Arial"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "SimHei"]
    rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    metrics = ["ARI", "NMI", "Purity", "V_Measure"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()
    for index, metric in enumerate(metrics):
        ax = axes[index]
        plot_data = []
        labels = []
        for method in METHOD_ORDER:
            vals = combined_df[combined_df["methods"] == method][metric].dropna().values
            if len(vals) > 0:
                plot_data.append(vals)
                labels.append(method)
        box = ax.boxplot(
            plot_data,
            vert=False,
            patch_artist=True,
            widths=0.6,
            medianprops={'color': 'black', 'linewidth': 1.5},
            whiskerprops={'color': 'black', 'linewidth': 1.2},
            capprops={'color': 'black', 'linewidth': 1.2},
            flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5},
        )
        for patch, method in zip(box['boxes'], labels):
            patch.set_facecolor(method_colors.get(method, '#808080'))
            patch.set_alpha(0.85)
            patch.set_linewidth(1.2)
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels)
        ax.set_title(metric, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.invert_yaxis()
    fig.subplots_adjust(
        left=0.30,
        right=0.98,
        bottom=0.08,
        top=0.94,
        wspace=0.32,
        hspace=0.28,
    )
    fig.align_ylabels(axes)
    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches="tight", dpi=300, facecolor='white')
    plt.close(fig)
    print(f"Saved boxplot figure to {output_path}")
def save_summary_table(ablation_dir: str, output_path: str) -> None:
    csv_files = sorted(
        [
            os.path.join(ablation_dir, name)
            for name in os.listdir(ablation_dir)
            if name.endswith(".csv")
        ]
    )
    if not csv_files:
        return
    combined_df = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
    combined_df = combined_df[combined_df["methods"].isin(METHOD_ORDER)].copy()
    combined_df["methods"] = pd.Categorical(
        combined_df["methods"], categories=METHOD_ORDER, ordered=True
    )
    rows = []
    for method in METHOD_ORDER:
        sub = combined_df[combined_df["methods"] == method]
        if sub.empty:
            continue
        row = {"Method": method}
        for m in ["ARI", "NMI", "Purity", "V_Measure"]:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_median"] = sub[m].median()
            row[f"{m}_std"] = sub[m].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    print(f"Saved summary table to {output_path}")
    print(summary.round(4).to_string(index=False))
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BRCA (Human_breast) ablation experiments.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_CONFIGS.keys()),
        choices=list(METHOD_CONFIGS.keys()),
        help="Method keys to run.",
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
    parser.add_argument(
        "--label-col",
        type=str,
        default="fine_annot_type",
        help="Ground truth label column in metadata.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=ABLATION_OUTPUT_DIR,
        help="Directory to write per-method CSVs and final boxplots.",
    )
    parser.add_argument(
        "--w-pro",
        type=float,
        default=DEFAULT_FIT_PARAMS["w_pro"],
        help="Override proxy-loss weight (w_pro). Default aligns with model_827_copy.py.",
    )
    parser.add_argument(
        "--w-info",
        type=float,
        default=DEFAULT_FIT_PARAMS["w_info"],
        help="Override InfoNCE loss weight (w_info). Default aligns with model_827_copy.py.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_FIT_PARAMS["lr"],
        help="Override learning rate.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    fit_params = dict(DEFAULT_FIT_PARAMS)
    fit_params["epochs"] = args.epochs
    fit_params["w_pro"] = args.w_pro
    fit_params["w_info"] = args.w_info
    fit_params["lr"] = args.lr
    # l_infonce/l_dgi method overrides are applied inside evaluate_single_run on top of fit_params
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in the current environment.")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    selected_methods = [METHOD_CONFIGS[key] for key in args.methods]
    print(f"Running methods ({len(selected_methods)}): {[m.display_name for m in selected_methods]}")
    print(f"Dataset: Human_breast, label_col={args.label_col}")
    print(f"Using device: {args.device}, epochs={fit_params['epochs']}")
    print(f"Fit params: {fit_params}")
    print(f"Output dir: {output_dir}")
    for method in selected_methods:
        print(f"\n{'=' * 60} {method.display_name} {'=' * 60}")
        try:
            result_df = evaluate_single_run(
                method=method,
                fit_params=fit_params,
                device=args.device,
                label_col=args.label_col,
                ablation_dir=output_dir,
            )
            save_method_results(method, result_df, output_dir)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Failed on {method.display_name}: {exc}")
            continue
    if not args.skip_plot:
        plot_path = os.path.join(output_dir, "BRCA_ablation_boxplots.png")
        create_boxplot_figure(output_dir, plot_path)
        summary_path = os.path.join(output_dir, "BRCA_ablation_summary.csv")
        save_summary_table(output_dir, summary_path)
if __name__ == "__main__":
    main()