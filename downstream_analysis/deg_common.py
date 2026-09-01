#!/usr/bin/env python
"""
Shared helpers for the DEG downstream analysis scripts
(run_brca_deg.py / run_pdac_deg.py).

This module contains every analysis step extracted from the downstream cells
of notebooks/dcpbst_BRAC.ipynb and notebooks/dcpbst_PDAC.ipynb:

  - Scanpy compatibility helpers (log1p metadata, rank_genes_groups extraction)
  - Wilcoxon rank-sum differential expression between two spatial domains
  - Web-platform export (genes x spots expression matrix + sample groups)
  - Top-10 DEG cluster-mean heatmap (hierarchical gene ordering, row colors)
  - Volcano plots (case perspective / reference perspective)
  - GO enrichment (hypergeometric test on the human GO annotation,
    Benjamini-Hochberg correction) + bubble plot

The notebooks only demonstrate spatial-domain identification; all DEG-related
reproduction lives in the two thin driver scripts that import this module.

Notes
-----
- No GPU / model weights are required: only saved clustered AnnData objects
  (saved_results/dcpbst_<DATASET>_adata_with_clusters.h5ad) are consumed.
- GO enrichment downloads goa_human.gaf.gz (human GOA annotation, ~15 MB) and
  go-basic.obo on first run; the annotation is cached once and shared by both
  scripts (downstream_analysis/deg_results/hsa_go_mapping.csv).
"""

# Auto-detect repo root (same convention as the other downstream scripts)
import os as _os
import sys as _sys

_CURRENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_CURRENT_DIR)
RESULTS_DIR = _os.path.join(REPO_ROOT, "saved_results")
_sys.path.insert(0, _os.path.join(REPO_ROOT, "dcpbst_package"))

import warnings

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved to disk, not displayed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import hypergeom

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global parameters (identical to the notebook cells)
# ---------------------------------------------------------------------------
CLUSTER_KEY = "domain"
LOGFC_THRESH = 2.0    # |log2FC| threshold
PVAL_THRESH = 0.05    # p-value threshold

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["figure.figsize"] = (12, 8)


# ---------------------------------------------------------------------------
# Scanpy compatibility helpers
# ---------------------------------------------------------------------------
def fix_log1p_metadata(adata):
    """Resolve the scanpy log1p 'base' KeyError across scanpy versions."""
    if "log1p" not in adata.uns:
        adata.uns["log1p"] = {}
    if "base" not in adata.uns["log1p"]:
        adata.uns["log1p"]["base"] = None
    return adata


def get_rank_genes_groups_df_compat(adata, group):
    """
    Extract rank_genes_groups results for one group into a DataFrame.
    Compatible with both dict and recarray storage used across Scanpy
    versions (avoids attribute errors).
    """
    rank_genes = adata.uns["rank_genes_groups"]

    genes = rank_genes["names"][group]
    logfc = rank_genes["logfoldchanges"][group]
    pvals = rank_genes["pvals"][group]

    pvals_adj = np.full_like(genes, np.nan, dtype=float)
    if isinstance(rank_genes, dict):
        if "pvals_adj" in rank_genes and group in rank_genes["pvals_adj"]:
            pvals_adj = rank_genes["pvals_adj"][group]
    else:  # recarray
        if "pvals_adj" in rank_genes.dtype.names:
            pvals_adj = rank_genes["pvals_adj"][group]

    de_df = pd.DataFrame(
        {
            "names": genes,
            "logfoldchanges": logfc,
            "pvals": pvals,
            "pvals_adj": pvals_adj,
        }
    )
    de_df = de_df.dropna(subset=["names", "logfoldchanges", "pvals"])
    de_df["names"] = de_df["names"].astype(str)
    return de_df


def run_deg(adata, case_group, ref_group, logfc_thresh=LOGFC_THRESH,
            pval_thresh=PVAL_THRESH):
    """Wilcoxon rank-sum DEG test: case_group vs ref_group."""
    print("\nRunning differential gene analysis "
          f"(cluster {case_group} vs cluster {ref_group})...")
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=CLUSTER_KEY,
            groups=[case_group],
            reference=ref_group,
            method="wilcoxon",
            log2fc_min=logfc_thresh,
            use_raw=False,
        )
        print("Differential gene analysis successful (use_raw=False)")
    except Exception as e:  # fallback for older/newer scanpy raw handling
        print(f"use_raw=False failed: {str(e)[:100]}; retrying use_raw=True...")
        sc.tl.rank_genes_groups(
            adata,
            groupby=CLUSTER_KEY,
            groups=[case_group],
            reference=ref_group,
            method="wilcoxon",
            log2fc_min=logfc_thresh,
            use_raw=True,
        )
        print("Differential gene analysis successful (use_raw=True)")

    de_df = get_rank_genes_groups_df_compat(adata, group=case_group)
    de_signif = de_df[
        (de_df["logfoldchanges"].abs() >= logfc_thresh)
        & (de_df["pvals"] < pval_thresh)
    ].sort_values("logfoldchanges", ascending=False).reset_index(drop=True)
    return de_df, de_signif


# ---------------------------------------------------------------------------
# Web-platform export (BRCA notebook cell 6)
# ---------------------------------------------------------------------------
def export_webtool_inputs(adata, groups, out_dir):
    """
    Export the genes x spots expression matrix (tab-separated) and the
    sample-group table for the online enrichment web platform.
    """
    adata_subset = adata[adata.obs[CLUSTER_KEY].isin(list(groups))].copy()
    print(f"\n[Web-platform export] cells after filtering: {adata_subset.n_obs}, "
          f"genes: {adata_subset.n_vars}")

    expr_matrix = adata_subset.X
    if sp.issparse(expr_matrix):
        expr_matrix = expr_matrix.T.toarray()  # genes x cells
    else:
        expr_matrix = expr_matrix.T

    heatmap_data = pd.DataFrame(
        expr_matrix,
        index=adata_subset.var_names,
        columns=adata_subset.obs_names,
    )
    heatmap_path = _os.path.join(out_dir, "webtool_data_heatmap.txt")
    heatmap_data.to_csv(heatmap_path, sep="\t", float_format="%.6f")

    sample_class = adata_subset.obs[[CLUSTER_KEY]].copy()
    sample_class.columns = ["Group"]
    sample_class.index.name = "Sample"
    class_path = _os.path.join(out_dir, "webtool_sample_class.txt")
    sample_class.to_csv(class_path, sep="\t")

    print(f"  Heatmap matrix saved: {heatmap_path} "
          f"({heatmap_data.shape[0]} genes x {heatmap_data.shape[1]} cells)")
    print(f"  Sample groups saved: {class_path}")
    for g in groups:
        print(f"  - cluster {g}: {(sample_class['Group'] == g).sum()} cells")


# ---------------------------------------------------------------------------
# Top-10 DEG heatmap
# ---------------------------------------------------------------------------
def plot_top10_heatmap(adata, de_signif, case_group, ref_group, save_path,
                       n_genes=10, show_legend=True):
    """
    Select 5 genes up-regulated in the case group and 5 up-regulated in the
    reference group, order genes hierarchically within each group, and draw
    the cluster-mean expression clustermap with row group colors.

    `de_signif` is already filtered to |log2FC| >= threshold, so splitting on
    logfoldchanges > 0 / < 0 is equivalent to the notebook's > 2 / < -2.
    """
    half = n_genes // 2
    up_case = de_signif[de_signif["logfoldchanges"] > 0].sort_values(
        "logfoldchanges", ascending=False)
    up_ref = de_signif[de_signif["logfoldchanges"] < 0].sort_values(
        "logfoldchanges", ascending=True)
    sel_up = up_case.head(half)["names"].tolist()
    sel_down = up_ref.head(half)["names"].tolist()

    # Pad with remaining genes if one direction has fewer than `half`
    remain = n_genes - (len(sel_up) + len(sel_down))
    if remain > 0:
        if len(sel_up) < half:
            sel_up += up_case.iloc[len(sel_up):].head(remain)["names"].tolist()
        if len(sel_down) < half:
            sel_down += up_ref.iloc[len(sel_down):].head(remain)["names"].tolist()

    top_genes = sel_up + sel_down
    print(f"  Heatmap genes: {len(sel_up)} up in {case_group}, "
          f"{len(sel_down)} up in {ref_group}, total {len(top_genes)}")

    # Mean expression of the top genes within every cluster
    expr_df = sc.get.obs_df(adata, keys=top_genes, use_raw=False)
    expr_df[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].values
    mean_mat = expr_df.groupby(CLUSTER_KEY).mean().T

    # Hierarchical ordering within the two gene groups (pairwise columns only)
    pair_cols = [c for c in mean_mat.columns if c in (case_group, ref_group)]
    pair_means = mean_mat[pair_cols].copy()
    row_group = pd.Series(index=pair_means.index, dtype=object)
    row_group.loc[sel_up] = case_group
    row_group.loc[sel_down] = ref_group
    row_group = row_group.fillna(
        (pair_means[case_group] >= pair_means[ref_group]).map(
            lambda x: case_group if x else ref_group)
    )

    order_indices = []
    for grp in (case_group, ref_group):
        idx = np.where(row_group.values == grp)[0]
        if len(idx) == 0:
            continue
        if len(idx) > 1:
            leaf = leaves_list(
                linkage(pair_means.values[idx], method="average",
                        metric="euclidean"))
            idx = idx[leaf]
        order_indices.extend(idx)

    ordered_genes = list(pair_means.index[order_indices])
    mean_mat_ordered = mean_mat.loc[ordered_genes]
    row_colors = row_group.loc[ordered_genes].map(
        {case_group: "#f39c12", ref_group: "#27ae60"})

    sns.set_context("notebook")
    g = sns.clustermap(
        mean_mat_ordered,
        cmap="viridis",
        metric="euclidean",
        method="average",
        row_cluster=False,
        col_cluster=True,
        figsize=(12, 8),
        xticklabels=True,
        yticklabels=True,
        row_colors=row_colors,
        cbar_pos=(0.52, 0.86, 0.02, 0.12),
        dendrogram_ratio=(0.05, 0.2),
    )

    if show_legend:
        handles = [
            mpatches.Patch(color="#f39c12", label=case_group),
            mpatches.Patch(color="#27ae60", label=ref_group),
        ]
        try:
            g.ax_row_dendrogram.legend(
                handles=handles,
                title="Row group",
                loc="center left",
                bbox_to_anchor=(-0.75, 0.5),
                frameon=False,
                fontsize=12,
                title_fontsize=10,
            )
        except Exception:
            pass

    # Reposition the colorbar and annotate it
    fig = g.fig
    cax = g.cax
    try:
        new_x, new_y, new_w, new_h = 0.98, 0.86, 0.015, 0.12
        cax.set_position([new_x, new_y, new_w, new_h])
        fig.text(
            new_x - 0.035, new_y + new_h / 2,
            "Mean expression\nin group",
            rotation=90, ha="left", va="center",
            fontsize=9, fontweight="normal", color="black",
        )
    except Exception as e:
        print(f"  Colorbar annotation failed: {e}")

    g.ax_heatmap.set_xlabel("Clusters", fontsize=12)
    g.ax_heatmap.set_ylabel("Genes", fontsize=12)
    plt.suptitle(
        f"Top 10 Differentially Expressed Genes ({case_group} vs {ref_group})",
        fontsize=16, fontweight="bold", x=0.65,
    )
    plt.tight_layout()
    g.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(g.fig)
    print(f"  Heatmap saved: {save_path}")


# ---------------------------------------------------------------------------
# Volcano plots
# ---------------------------------------------------------------------------
def plot_volcano(de_df, case_group, ref_group, logfc_thresh, pval_thresh,
                 save_path, annotate_fontsize=9, with_title=True):
    """
    Volcano plot from the case-group perspective. Up/down labels refer to the
    case group; top-10 genes by |log2FC| * -log10(P) are annotated.
    """
    de_df = de_df.copy()
    de_df["neg_log10_p"] = -np.log10(de_df["pvals"] + 1e-10)
    de_df["category"] = "Non-significant"
    de_df.loc[
        (de_df["logfoldchanges"] >= logfc_thresh) & (de_df["pvals"] < pval_thresh),
        "category",
    ] = f"Up-regulated ({case_group})"
    de_df.loc[
        (de_df["logfoldchanges"] <= -logfc_thresh) & (de_df["pvals"] < pval_thresh),
        "category",
    ] = f"Down-regulated ({case_group})"

    color_map = {
        f"Up-regulated ({case_group})": "#e74c3c",
        f"Down-regulated ({case_group})": "#3498db",
        "Non-significant": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    for cat, color in color_map.items():
        mask = de_df["category"] == cat
        ax.scatter(
            de_df.loc[mask, "logfoldchanges"],
            de_df.loc[mask, "neg_log10_p"],
            c=color, alpha=0.6, s=30,
            label=f"{cat} (n={mask.sum()})",
        )

    ax.axvline(x=logfc_thresh, color="black", linestyle="--", alpha=0.5, lw=1)
    ax.axvline(x=-logfc_thresh, color="black", linestyle="--", alpha=0.5, lw=1)
    ax.axhline(y=-np.log10(pval_thresh), color="black", linestyle="--",
               alpha=0.5, lw=1)

    de_df["score"] = de_df["logfoldchanges"].abs() * de_df["neg_log10_p"]
    for _, row in de_df.nlargest(10, "score").iterrows():
        ax.annotate(
            row["names"],
            xy=(row["logfoldchanges"], row["neg_log10_p"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=annotate_fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel(
        f"log2(Fold Change) (Cluster {case_group} vs {ref_group})",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("-log10(P-value)", fontsize=14, fontweight="bold")
    if with_title:
        ax.set_title(
            f"Volcano Plot of Differential Expressed Genes\n"
            f"(Cluster {case_group} vs {ref_group}, "
            f"|log2FC|>={logfc_thresh:g}, P<{pval_thresh:g})",
            fontsize=16, fontweight="bold", pad=20,
        )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Volcano plot saved: {save_path}")
    print(f"    Up-regulated in {case_group}: "
          f"{(de_df['category'] == f'Up-regulated ({case_group})').sum()}")
    print(f"    Down-regulated in {case_group}: "
          f"{(de_df['category'] == f'Down-regulated ({case_group})').sum()}")
    print(f"    Non-significant: "
          f"{(de_df['category'] == 'Non-significant').sum()}")


def plot_volcano_reference_based(de_df, case_group, ref_group, logfc_thresh,
                                 pval_thresh, save_path, top_n_up=5):
    """
    Volcano plot from the reference-group perspective (BRCA notebook cell 12).
    The `top_n_up` most significant genes up-regulated in the REFERENCE group
    are annotated with a highlighted box; 5 other significant genes are also
    labelled.
    """
    de_df = de_df.copy()
    de_df["neg_log10_p"] = -np.log10(de_df["pvals"] + 1e-10)
    de_df["category"] = "Non-significant"
    de_df.loc[
        (de_df["logfoldchanges"] <= -logfc_thresh) & (de_df["pvals"] < pval_thresh),
        "category",
    ] = f"Up-regulated ({ref_group})"
    de_df.loc[
        (de_df["logfoldchanges"] >= logfc_thresh) & (de_df["pvals"] < pval_thresh),
        "category",
    ] = f"Down-regulated ({ref_group})"

    color_map = {
        f"Up-regulated ({ref_group})": "#3498db",
        f"Down-regulated ({ref_group})": "#e74c3c",
        "Non-significant": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    for cat, color in color_map.items():
        mask = de_df["category"] == cat
        ax.scatter(
            de_df.loc[mask, "logfoldchanges"],
            de_df.loc[mask, "neg_log10_p"],
            c=color, alpha=0.6, s=30,
            label=f"{cat} (n={mask.sum()})",
        )

    ax.axvline(x=logfc_thresh, color="black", linestyle="--", alpha=0.5, lw=1)
    ax.axvline(x=-logfc_thresh, color="black", linestyle="--", alpha=0.5, lw=1)
    ax.axhline(y=-np.log10(pval_thresh), color="black", linestyle="--",
               alpha=0.5, lw=1)

    ref_up = de_df[de_df["category"] == f"Up-regulated ({ref_group})"].sort_values(
        "pvals", ascending=True)
    top_ref_up = ref_up.head(top_n_up)
    print(f"  Annotating top {len(top_ref_up)} genes up-regulated in "
          f"{ref_group}: {top_ref_up['names'].tolist()}")
    for _, row in top_ref_up.iterrows():
        ax.annotate(
            row["names"],
            xy=(row["logfoldchanges"], row["neg_log10_p"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue",
                      alpha=0.8),
            color="darkblue",
        )

    other = de_df[de_df["category"] != f"Up-regulated ({ref_group})"].sort_values(
        "neg_log10_p", ascending=False)
    for _, row in other.head(5).iterrows():
        ax.annotate(
            row["names"],
            xy=(row["logfoldchanges"], row["neg_log10_p"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel(
        f"log2(Fold Change) (Cluster {case_group} / Cluster {ref_group})",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("-log10(P-value)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Volcano Plot of Differential Expressed Genes\n"
        f"(Reference: Cluster {ref_group}, "
        f"|log2FC|>={logfc_thresh:g}, P<{pval_thresh:g})",
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Reference-based volcano plot saved: {save_path}")


# ---------------------------------------------------------------------------
# GO enrichment (hypergeometric test + BH correction)
# ---------------------------------------------------------------------------
def download_go_mapping(cache_path):
    """Download the human GO annotation (GOA human GAF) and parse gene-GO mapping.

    GAF 2.2 fields (0-indexed): 2 = DB Object Symbol (gene symbol),
    4 = GO ID, 8 = Aspect (P=BP, C=CC, F=MF).
    """
    import io
    import requests

    gaf_urls = [
        "https://current.geneontology.org/annotations/goa_human.gaf.gz",
        "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz",
    ]
    last_err = None
    for url in gaf_urls:
        try:
            print(f"Downloading human GO annotation from {url} ...")
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            go_df = pd.read_csv(
                io.BytesIO(response.content),
                sep="\t",
                comment="!",
                header=None,
                compression="gzip",
                usecols=[2, 4, 8],
                names=["gene_symbol", "go_id", "go_category"],
            )
            break
        except Exception as e:  # try next mirror
            last_err = e
            print(f"  download failed: {str(e)[:120]}")
    else:
        raise RuntimeError(
            "Failed to download GO annotation. Please download goa_human.gaf.gz "
            "manually from geneontology.org, then re-run.") from last_err

    go_df = go_df.dropna()
    go_df = go_df[go_df["go_category"].isin(["P", "C", "F"])]
    go_df["go_category"] = go_df["go_category"].map(
        {"P": "BP", "C": "CC", "F": "MF"})
    go_df.to_csv(cache_path, index=False)
    print(f"GO mapping cached: {cache_path}")
    return go_df


def load_go_mapping(cache_path):
    """Load gene-GO mapping from cache, downloading on first use."""
    if _os.path.exists(cache_path):
        print(f"Loading cached GO mapping: {cache_path}")
        return pd.read_csv(cache_path)
    return download_go_mapping(cache_path)


def load_go_terms():
    """Parse go-basic.obo into a {GO id: term name} dictionary."""
    import requests

    go_term_url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    print("Loading GO term names...")
    response = requests.get(go_term_url, stream=True)
    response.raise_for_status()
    go_terms = {}
    current_id = None
    current_name = None
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("id: GO:"):
            current_id = line.split(" ")[1]
        elif line.startswith("name: "):
            current_name = line[6:].strip()
        elif line == "":
            if current_id and current_name:
                go_terms[current_id] = current_name
            current_id = None
            current_name = None
    print(f"Loaded {len(go_terms)} GO terms")
    return go_terms


def run_go_enrichment(signif_genes, out_dir, prefix, cache_path=None):
    """
    Hypergeometric GO enrichment with Benjamini-Hochberg correction.
    Returns the significant GO terms DataFrame (or None).

    `cache_path` points at the shared gene-GO annotation cache; it defaults to
    <out_dir>/hsa_go_mapping.csv.
    """
    if len(signif_genes) == 0:
        raise ValueError("No significant DEGs; cannot run GO enrichment.")

    if cache_path is None:
        cache_path = _os.path.join(out_dir, "hsa_go_mapping.csv")
    go_mapping = load_go_mapping(cache_path)
    go_terms = load_go_terms()

    background_genes = set(go_mapping["gene_symbol"].unique())
    print(f"  Background genes with GO annotation: {len(background_genes)}")
    print(f"  Significant DEGs: {len(signif_genes)}")
    signif_with_go = [g for g in signif_genes if g in background_genes]
    if len(signif_with_go) == 0:
        raise ValueError("No significant DEGs have GO annotation.")
    print(f"  Significant DEGs with GO annotation: {len(signif_with_go)}")

    enrichment = []
    for go_id, group in go_mapping.groupby("go_id"):
        go_background = set(group["gene_symbol"].unique())
        M = len(background_genes)
        n = len(go_background)
        N = len(signif_with_go)
        k = len(go_background & set(signif_with_go))
        if k == 0:
            continue
        p_val = hypergeom.sf(k - 1, M, n, N)
        fe = (k / N) / (n / M)
        enrichment.append({
            "go_id": go_id,
            "go_term": go_terms.get(go_id, "Unknown"),
            "go_category": group["go_category"].iloc[0],
            "background_count": n,
            "significant_count": k,
            "fold_enrichment": fe,
            "p_val": p_val,
        })

    results_df = pd.DataFrame(enrichment)
    if results_df.empty:
        print("  No enriched GO terms found.")
        return None

    # Benjamini-Hochberg correction
    results_df = results_df.sort_values("p_val")
    results_df["adj_p_val"] = (
        results_df["p_val"] * len(results_df) / np.arange(1, len(results_df) + 1)
    )
    results_df["adj_p_val"] = results_df["adj_p_val"].clip(upper=1.0)

    results_df.to_csv(_os.path.join(out_dir, f"{prefix}_all_results.csv"),
                      index=False)

    significant = results_df[results_df["adj_p_val"] < 0.05].copy()
    if significant.empty:
        print("  No significantly enriched GO terms (adj_p_val < 0.05).")
        return None

    cat_label = {"BP": "Biological Process", "CC": "Cellular Component",
                 "MF": "Molecular Function"}
    significant["go_category_name"] = significant["go_category"].map(cat_label)
    significant["neg_log10_adjp"] = -np.log10(significant["adj_p_val"])
    significant.to_csv(_os.path.join(out_dir, f"{prefix}_significant.csv"),
                       index=False)

    print(f"  Significant GO terms: {len(significant)} "
          f"(BP: {(significant['go_category'] == 'BP').sum()}, "
          f"CC: {(significant['go_category'] == 'CC').sum()}, "
          f"MF: {(significant['go_category'] == 'MF').sum()})")
    return significant


def plot_go_bubble(go_results, save_path, case_group, ref_group):
    """Bubble plot of top-10 significant GO terms per category (BP/CC/MF)."""
    if go_results is None or go_results.empty:
        print("  No significant enrichment; bubble plot skipped.")
        return

    top_go = []
    for cat in ["BP", "CC", "MF"]:
        cat_data = go_results[go_results["go_category"] == cat].copy()
        if len(cat_data) > 0:
            top_go.append(cat_data.nlargest(min(10, len(cat_data)),
                                           "neg_log10_adjp"))
    top_go_df = pd.concat(top_go, ignore_index=True)

    cat_color = {"BP": "#e74c3c", "CC": "#3498db", "MF": "#2ecc71"}
    cat_label = {"BP": "Biological Process", "CC": "Cellular Component",
                 "MF": "Molecular Function"}

    fig, ax = plt.subplots(figsize=(12, 10))
    for cat, color in cat_color.items():
        mask = top_go_df["go_category"] == cat
        if mask.sum() == 0:
            continue
        sizes = top_go_df.loc[mask, "significant_count"] * 15
        ax.scatter(
            top_go_df.loc[mask, "neg_log10_adjp"],
            top_go_df.loc[mask, "go_term"],
            s=sizes, c=color, alpha=0.7,
            edgecolors="black", linewidth=0.5, label=cat_label[cat],
        )

    ax.axvline(x=-np.log10(0.05), color="black", linestyle="--", alpha=0.5,
               lw=1)
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=14, fontweight="bold")
    ax.set_ylabel("GO Terms", fontsize=14, fontweight="bold")
    ax.set_title(
        f"GO Enrichment Analysis (Cluster {case_group} vs {ref_group})\n"
        "Top 10 Terms per Category",
        fontsize=16, fontweight="bold", pad=20,
    )
    cat_legend = [mpatches.Patch(color=color, alpha=0.7, label=cat_label[cat])
                  for cat, color in cat_color.items()]
    size_legend = [mpatches.Patch(color="gray", alpha=0.7, label=f"{n} genes")
                   for n in [2, 5, 10, 20]]
    ax.legend(handles=cat_legend + size_legend, loc="upper right", fontsize=10)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  GO bubble plot saved: {save_path}")
