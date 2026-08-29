# Auto-detect repo root for reproducibility
import os as _os, sys as _sys
_CURRENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_CURRENT_DIR)
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = _os.path.join(REPO_ROOT, 'saved_results')
_sys.path.insert(0, _os.path.join(REPO_ROOT, 'dcpbst_package'))
# #!/usr/bin/env python3
# """
# Export significant differential gene IDs (one-per-line .txt) for Metascape upload,
# replicating the logic in tutorial_dlpfc_brca_go_svg.ipynb (lines 1-222).
# """
# import argparse
# import os
# from typing import Tuple
# import numpy as np
# import pandas as pd
# import scanpy as sc
# def fix_log1p_metadata(adata):
#     if 'log1p' not in adata.uns:
#         adata.uns['log1p'] = {}
#     if 'base' not in adata.uns['log1p']:
#         adata.uns['log1p']['base'] = None
#     return adata
# def get_rank_genes_groups_df_compat(adata, group: str) -> pd.DataFrame:
#     """
#     Robustly extract rank_genes_groups into a DataFrame with columns:
#     names, logfoldchanges, pvals, pvals_adj (may be NaN).
#     Compatible with dict/recarray variants used across Scanpy versions.
#     """
#     rank_genes = adata.uns['rank_genes_groups']
#     # mandatory fields
#     genes = rank_genes['names'][group]
#     logfc = rank_genes['logfoldchanges'][group]
#     pvals = rank_genes['pvals'][group]
#     # optional adjusted p-values
#     pvals_adj = np.full_like(genes, np.nan, dtype=float)
#     if isinstance(rank_genes, dict):
#         if 'pvals_adj' in rank_genes and group in rank_genes['pvals_adj']:
#             pvals_adj = rank_genes['pvals_adj'][group]
#     else:
#         if 'pvals_adj' in rank_genes.dtype.names:
#             pvals_adj = rank_genes['pvals_adj'][group]
#     de_df = pd.DataFrame(
#         {
#             'names': genes,
#             'logfoldchanges': logfc,
#             'pvals': pvals,
#             'pvals_adj': pvals_adj,
#         }
#     )
#     de_df = de_df.dropna(subset=['names', 'logfoldchanges', 'pvals'])
#     de_df['names'] = de_df['names'].astype(str)
#     return de_df
# def run_rank_genes_groups(
#     adata,
#     cluster_key: str,
#     case_group: str,
#     ref_group: str,
#     log2fc_min: float,
# ) -> None:
#     """
#     Execute Scanpy rank_genes_groups with a fallback between use_raw False/True.
#     """
#     try:
#         sc.tl.rank_genes_groups(
#             adata,
#             groupby=cluster_key,
#             groups=[case_group],
#             reference=ref_group,
#             method='wilcoxon',
#             log2fc_min=log2fc_min,
#             use_raw=False,
#         )
#     except Exception:
#         sc.tl.rank_genes_groups(
#             adata,
#             groupby=cluster_key,
#             groups=[case_group],
#             reference=ref_group,
#             method='wilcoxon',
#             log2fc_min=log2fc_min,
#             use_raw=True,
#         )
# def export_for_metascape(
#     adata_path: str,
#     output_dir: str,
#     cluster_key: str = 'domain',
#     case_group: str = '12',
#     ref_group: str = '14',
#     log2fc_thresh: float = 2.0,
#     pval_thresh: float = 0.05,
# ) -> Tuple[str, str]:
#     """
#     Produce:
#     - TXT with one gene ID per line (ready for Metascape)
#     - TSV with names, log2FC, pval, pval_adj (for record)
#     Returns (txt_path, tsv_path).
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     adata = sc.read(adata_path)
#     adata = fix_log1p_metadata(adata)
#     run_rank_genes_groups(
#         adata,
#         cluster_key=cluster_key,
#         case_group=case_group,
#         ref_group=ref_group,
#         log2fc_min=log2fc_thresh,
#     )
#     de_df = get_rank_genes_groups_df_compat(adata, group=case_group)
#     de_signif = de_df[
#         (de_df['logfoldchanges'].abs() >= log2fc_thresh) & (de_df['pvals'] < pval_thresh)
#     ].copy()
#     # Deduplicate while preserving original order
#     unique_genes = pd.Series(de_signif['names']).drop_duplicates().tolist()
#     base = f"deg_{case_group}_vs_{ref_group}"
#     txt_path = os.path.join(output_dir, f"{base}_genes.txt")
#     tsv_path = os.path.join(output_dir, f"{base}_with_stats.tsv")
#     # One ID per line for Metascape upload
#     with open(txt_path, 'w', encoding='utf-8') as f:
#         for g in unique_genes:
#             f.write(f"{g}\n")
#     # Save stats for transparency
#     de_signif[['names', 'logfoldchanges', 'pvals', 'pvals_adj']].to_csv(
#         tsv_path, sep='\t', index=False
#     )
#     return txt_path, tsv_path
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Export significant DE genes (IDs) for Metascape upload."
#     )
#     p.add_argument(
#         "--adata",
#         default=os.path.join(RESULTS_DIR, "miso_adata_with_clusters.h5ad"),
#         help="Path to input AnnData (.h5ad).",
#     )
#     p.add_argument(
#         "--cluster-key",
#         default="domain",
#         help="Obs column name used for clustering/groups.",
#     )
#     p.add_argument("--case", default="8", help="Case group label (e.g., '8').")
#     p.add_argument("--ref", default="12", help="Reference group label (e.g., '12').")
#     p.add_argument(
#         "--log2fc",
#         type=float,
#         default=2.0,
#         help="Absolute log2 fold-change threshold (|log2FC| >= this).",
#     )
#     p.add_argument(
#         "--pval",
#         type=float,
#         default=0.05,
#         help="P-value threshold (p < this).",
#     )
#     p.add_argument(
#         "--outdir",
#         default=os.path.join(REPO_ROOT, "exports"),
#         help="Output directory for exported files.",
#     )
#     return p.parse_args()
# def main() -> None:
#     args = parse_args()
#     txt_path, tsv_path = export_for_metascape(
#         adata_path=args.adata,
#         output_dir=args.outdir,
#         cluster_key=args.cluster_key,
#         case_group=args.case,
#         ref_group=args.ref,
#         log2fc_thresh=args.log2fc,
#         pval_thresh=args.pval,
#     )
#     print("Export completed.")
#     print(f"Metascape gene list: {txt_path}")
#     print(f"With stats (TSV):    {tsv_path}")
#     print(
#         "Tip: Upload the TXT (one ID per line) to Metascape. "
#         "Avoid opening in Excel to prevent gene symbol auto-conversion."
#     )
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Export significant differential gene IDs (one-per-line .txt) for Metascape upload,
replicating the logic in tutorial_dlpfc_brca_go_svg.ipynb (lines 1-222).
"""
import argparse
import os
from typing import Tuple
import numpy as np
import pandas as pd
import scanpy as sc
def fix_log1p_metadata(adata):
    if 'log1p' not in adata.uns:
        adata.uns['log1p'] = {}
    if 'base' not in adata.uns['log1p']:
        adata.uns['log1p']['base'] = None
    return adata
def get_rank_genes_groups_df_compat(adata, group: str) -> pd.DataFrame:
    """
    Robustly extract rank_genes_groups into a DataFrame with columns:
    names, logfoldchanges, pvals, pvals_adj (may be NaN).
    Compatible with dict/recarray variants used across Scanpy versions.
    """
    rank_genes = adata.uns['rank_genes_groups']
    # mandatory fields
    genes = rank_genes['names'][group]
    logfc = rank_genes['logfoldchanges'][group]
    pvals = rank_genes['pvals'][group]
    # optional adjusted p-values
    pvals_adj = np.full_like(genes, np.nan, dtype=float)
    if isinstance(rank_genes, dict):
        if 'pvals_adj' in rank_genes and group in rank_genes['pvals_adj']:
            pvals_adj = rank_genes['pvals_adj'][group]
    else:
        if 'pvals_adj' in rank_genes.dtype.names:
            pvals_adj = rank_genes['pvals_adj'][group]
    de_df = pd.DataFrame(
        {
            'names': genes,
            'logfoldchanges': logfc,
            'pvals': pvals,
            'pvals_adj': pvals_adj,
        }
    )
    de_df = de_df.dropna(subset=['names', 'logfoldchanges', 'pvals'])
    de_df['names'] = de_df['names'].astype(str)
    return de_df
def run_rank_genes_groups(
    adata,
    cluster_key: str,
    case_group: str,
    ref_group: str,
    log2fc_min: float,
) -> None:
    """
    Execute Scanpy rank_genes_groups with a fallback between use_raw False/True.
    """
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            groups=[case_group],
            reference=ref_group,
            method='wilcoxon',
            log2fc_min=log2fc_min,
            use_raw=False,
        )
    except Exception:
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            groups=[case_group],
            reference=ref_group,
            method='wilcoxon',
            log2fc_min=log2fc_min,
            use_raw=True,
        )
def export_for_metascape(
    adata_path: str,
    output_dir: str,
    cluster_key: str = 'domain',
    case_group: str = '3',
    ref_group: str = '4',
    log2fc_thresh: float = 2.0,
    pval_thresh: float = 0.05,
) -> Tuple[str, str]:
    """
    Produce:
    - TXT with one gene ID per line (ready for Metascape)
    - TSV with names, log2FC, pval, pval_adj (for record)
    Returns (txt_path, tsv_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    adata = sc.read(adata_path)
    adata = fix_log1p_metadata(adata)
    run_rank_genes_groups(
        adata,
        cluster_key=cluster_key,
        case_group=case_group,
        ref_group=ref_group,
        log2fc_min=log2fc_thresh,
    )
    de_df = get_rank_genes_groups_df_compat(adata, group=case_group)
    de_signif = de_df[
        (de_df['logfoldchanges'].abs() >= log2fc_thresh) & (de_df['pvals'] < pval_thresh)
    ].copy()
    # Deduplicate while preserving original order
    unique_genes = pd.Series(de_signif['names']).drop_duplicates().tolist()
    base = f"deg_{case_group}_vs_{ref_group}"
    txt_path = os.path.join(output_dir, f"{base}_genes_pdac.txt")
    tsv_path = os.path.join(output_dir, f"{base}_with_stats_pdac.tsv")
    # One ID per line for Metascape upload
    with open(txt_path, 'w', encoding='utf-8') as f:
        for g in unique_genes:
            f.write(f"{g}\n")
    # Save stats for transparency
    de_signif[['names', 'logfoldchanges', 'pvals', 'pvals_adj']].to_csv(
        tsv_path, sep='\t', index=False
    )
    return txt_path, tsv_path
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export significant DE genes (IDs) for Metascape upload."
    )
    p.add_argument(
        "--adata",
        default=os.path.join(RESULTS_DIR, "miso_adata_with_clusters.h5ad"),
        help="Path to input AnnData (.h5ad).",
    )
    p.add_argument(
        "--cluster-key",
        default="domain",
        help="Obs column name used for clustering/groups.",
    )
    p.add_argument("--case", default="3", help="Case group label (e.g., '3').")
    p.add_argument("--ref", default="4", help="Reference group label (e.g., '4').")
    p.add_argument(
        "--log2fc",
        type=float,
        default=2.0,
        help="Absolute log2 fold-change threshold (|log2FC| >= this).",
    )
    p.add_argument(
        "--pval",
        type=float,
        default=0.05,
        help="P-value threshold (p < this).",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join(REPO_ROOT, "exports"),
        help="Output directory for exported files.",
    )
    return p.parse_args()
def main() -> None:
    args = parse_args()
    txt_path, tsv_path = export_for_metascape(
        adata_path=args.adata,
        output_dir=args.outdir,
        cluster_key=args.cluster_key,
        case_group=args.case,
        ref_group=args.ref,
        log2fc_thresh=args.log2fc,
        pval_thresh=args.pval,
    )
    print("Export completed.")
    print(f"Metascape gene list: {txt_path}")
    print(f"With stats (TSV):    {tsv_path}")
    print(
        "Tip: Upload the TXT (one ID per line) to Metascape. "
        "Avoid opening in Excel to prevent gene symbol auto-conversion."
    )
if __name__ == "__main__":
    main()