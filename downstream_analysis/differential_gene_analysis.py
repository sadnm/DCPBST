import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu, hypergeom
import seaborn as sns
import os
import warnings
import random
import torch
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 100
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.figsize'] = (12, 8)

cluster_key = 'domain'
logfc_thresh = 2
pval_thresh = 0.05

def fix_log1p_metadata(adata):
    if 'log1p' not in adata.uns:
        adata.uns['log1p'] = {}
    if 'base' not in adata.uns['log1p']:
        adata.uns['log1p']['base'] = None
    return adata

def get_rank_genes_groups_df_compat(adata, group):
    rank_genes = adata.uns['rank_genes_groups']
    if isinstance(rank_genes, dict):
        genes = rank_genes['names'][group]
        logfc = rank_genes['logfoldchanges'][group]
        pvals = rank_genes['pvals'][group]
    else:
        genes = rank_genes['names'][group]
        logfc = rank_genes['logfoldchanges'][group]
        pvals = rank_genes['pvals'][group]
    pvals_adj = np.full_like(genes, np.nan, dtype=float)
    if isinstance(rank_genes, dict):
        if 'pvals_adj' in rank_genes:
            try:
                pvals_adj = rank_genes['pvals_adj'][group]
            except:
                pass
    else:
        try:
            pvals_adj = rank_genes['pvals_adj'][group]
        except:
            pass
    df = pd.DataFrame({
        'names': genes,
        'logfoldchanges': logfc,
        'pvals': pvals,
        'pvals_adj': pvals_adj
    })
    return df

def run_differential_analysis(adata_path, output_dir='./deg_results'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading adata from: {adata_path}")
    adata = sc.read(adata_path)
    adata = fix_log1p_metadata(adata)
    print(f"adata shape: {adata.shape}")
    print(f"Clusters in '{cluster_key}':", adata.obs[cluster_key].unique().tolist())
    
    print("\nRunning Wilcoxon rank-sum test for differential expression...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method='wilcoxon',
        corr_method='benjamini-hochberg',
        log2fc_threshold=0.0,
        min_in_group_fraction=0.0
    )
    
    all_degs = []
    for cluster_id in adata.obs[cluster_key].unique():
        deg_df = get_rank_genes_groups_df_compat(adata, cluster_id)
        deg_df['cluster'] = cluster_id
        deg_df['is_significant'] = (
            (deg_df['pvals_adj'].fillna(deg_df['pvals']) < pval_thresh) &
            (deg_df['logfoldchanges'].abs() >= logfc_thresh)
        )
        sig_df = deg_df[deg_df['is_significant']].copy()
        sig_up = sig_df[sig_df['logfoldchanges'] > 0]
        sig_down = sig_df[sig_df['logfoldchanges'] < 0]
        print(f"Cluster {cluster_id}: {len(sig_df)} DEGs (UP: {len(sig_up)}, DOWN: {len(sig_down)})")
        deg_df.to_csv(f"{output_dir}/DEG_cluster_{cluster_id}_all.csv", index=False)
        sig_df.to_csv(f"{output_dir}/DEG_cluster_{cluster_id}_significant.csv", index=False)
        all_degs.append(deg_df)
    
    all_degs_df = pd.concat(all_degs, ignore_index=True)
    all_degs_df.to_csv(f"{output_dir}/DEG_all_clusters_all_genes.csv", index=False)
    
    print("\nPlotting ranked genes groups...")
    try:
        sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
        plt.savefig(f"{output_dir}/rank_genes_groups_top25.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: rank_genes_groups plot failed: {e}")
    
    try:
        sc.pl.rank_genes_groups_violin(adata, groups=adata.obs[cluster_key].unique()[:5], n_genes=8, show=False)
        plt.savefig(f"{output_dir}/rank_genes_groups_violin.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: violin plot failed: {e}")
    
    print(f"\nAll DEG results saved to {output_dir}/")
    return adata, all_degs_df

def run_mannwhitney_pairwise(adata_path, output_dir='./deg_results'):
    os.makedirs(output_dir, exist_ok=True)
    adata = sc.read(adata_path)
    adata = fix_log1p_metadata(adata)
    
    clusters = adata.obs[cluster_key].unique()
    print(f"\nRunning pairwise Mann-Whitney U test for {len(clusters)} clusters...")
    
    results = []
    gene_names = adata.var_names.tolist()
    from itertools import combinations
    for c1, c2 in combinations(clusters, 2):
        idx1 = adata.obs[cluster_key] == c1
        idx2 = adata.obs[cluster_key] == c2
        if idx1.sum() < 2 or idx2.sum() < 2:
            continue
        X1 = adata[idx1].X.toarray() if hasattr(adata[idx1].X, 'toarray') else adata[idx1].X
        X2 = adata[idx2].X.toarray() if hasattr(adata[idx2].X, 'toarray') else adata[idx2].X
        for g_idx, gene in enumerate(gene_names[:200]):
            try:
                stat, pval = mannwhitneyu(X1[:, g_idx], X2[:, g_idx], alternative='two-sided')
                logfc = np.log2((X1[:, g_idx].mean() + 1e-9) / (X2[:, g_idx].mean() + 1e-9))
                results.append({
                    'cluster_1': c1, 'cluster_2': c2,
                    'gene': gene, 'U_statistic': stat,
                    'p_value': pval, 'log2FC': logfc
                })
            except:
                pass
    
    pairwise_df = pd.DataFrame(results)
    if len(pairwise_df) > 0:
        pairwise_df['adj_p_value'] = pairwise_df.groupby(['cluster_1', 'cluster_2'])['p_value'].transform(
            lambda x: np.minimum(x * len(x), 1.0)
        )
        pairwise_df.to_csv(f"{output_dir}/pairwise_MWU_DEGs_top200genes.csv", index=False)
        print(f"Pairwise results: {len(pairwise_df)} tests saved.")
    return pairwise_df

if __name__ == "__main__":
    import sys
    adata_path = sys.argv[1] if len(sys.argv) > 1 else None
    if adata_path is None:
        print("Usage: python differential_gene_analysis.py <path_to_clustered_adata.h5ad>")
        print("Example: python differential_gene_analysis.py saved_results/dcpbst_adata_with_clusters.h5ad")
        sys.exit(1)
    run_differential_analysis(adata_path)
    run_mannwhitney_pairwise(adata_path)
