#!/usr/bin/env python
"""
PDAC (pancreatic ductal adenocarcinoma, GSE111672) downstream analysis:
differential expression.

Standalone reviewer-facing driver for the DEG downstream tasks of
notebooks/dcpbst_PDAC.ipynb. The notebook itself only demonstrates
spatial-domain identification (clustering + ARI evaluation); every downstream
task related to differentially expressed genes (DEGs) lives here:

  1. Export of the raw genes x spots expression matrix and sample-group table
     for the online enrichment web platform (domains 3 and 4)
  2. Wilcoxon DEG test between two spatial domains (clusters 3 vs 4),
     full and significant DEG tables exported as CSV
  3. Top-10 DEG cluster-mean heatmap (5 up / 5 down, hierarchical ordering
     with row group colors)
  4. Volcano plot (annotate top-10 by |log2FC|*-log10P)
  5. GO enrichment (hypergeometric test on the human GO annotation,
     Benjamini-Hochberg correction) + bubble plot

All analysis functions are shared with run_brca_deg.py via deg_common.py.

Input
-----
saved_results/dcpbst_PDAC_adata_with_clusters.h5ad
    Produced by running the spatial-domain identification pipeline in
    notebooks/dcpbst_PDAC.ipynb (adata with adata.obs['domain']).

Usage
-----
    python downstream_analysis/run_pdac_deg.py
    python downstream_analysis/run_pdac_deg.py --skip-go        # offline
    python downstream_analysis/run_pdac_deg.py --skip-go --skip-webtool
    python downstream_analysis/run_pdac_deg.py \
        --adata saved_results/dcpbst_PDAC_adata_with_clusters.h5ad \
        --outdir downstream_analysis/deg_results/pdac

Notes
-----
- No GPU / model weights are required: this script only consumes the saved
  clustered AnnData.
- The web-platform files (webtool_data_heatmap.txt / webtool_sample_class.txt)
  are the single raw gene-expression export kept for reviewers; they can be
  uploaded to the online enrichment platform.
- GO enrichment downloads goa_human.gaf.gz (human GOA annotation, ~15 MB) and
  go-basic.obo on first run; the annotation is cached once and shared with the
  BRCA script under downstream_analysis/deg_results/. Use --skip-go offline.
"""

import os as _os

import scanpy as sc

from deg_common import (
    CLUSTER_KEY,
    LOGFC_THRESH,
    PVAL_THRESH,
    RESULTS_DIR,
    export_webtool_inputs,
    fix_log1p_metadata,
    plot_go_bubble,
    plot_top10_heatmap,
    plot_volcano,
    run_deg,
    run_go_enrichment,
)

# ---------------------------------------------------------------------------
# Dataset-specific parameters (identical to the notebook cells)
# ---------------------------------------------------------------------------
CASE_GROUP = "3"      # case spatial domain
REF_GROUP = "4"       # reference spatial domain
WEBTOOL_GROUPS = ("3", "4")  # domains exported for the web platform
DEFAULT_OUTDIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "deg_results", "pdac")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PDAC DEG downstream analysis (notebooks/dcpbst_PDAC.ipynb).")
    parser.add_argument(
        "--adata",
        default=_os.path.join(RESULTS_DIR,
                              "dcpbst_PDAC_adata_with_clusters.h5ad"),
        help="Path to the clustered AnnData produced by the PDAC notebook.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Directory for all generated tables and figures.",
    )
    parser.add_argument("--case", default=CASE_GROUP, help="Case domain label.")
    parser.add_argument("--ref", default=REF_GROUP,
                        help="Reference domain label.")
    parser.add_argument("--logfc-thresh", type=float, default=LOGFC_THRESH)
    parser.add_argument("--pval-thresh", type=float, default=PVAL_THRESH)
    parser.add_argument(
        "--skip-webtool", action="store_true",
        help="Skip export of the web-platform input files (clusters 3 and 4).",
    )
    parser.add_argument(
        "--skip-go", action="store_true",
        help="Skip GO enrichment (no network access required).",
    )
    args = parser.parse_args()

    _os.makedirs(args.outdir, exist_ok=True)
    print(f"Repo root:  {_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}")
    print(f"Loading adata: {args.adata}")
    adata = sc.read_h5ad(args.adata)
    adata = fix_log1p_metadata(adata)
    print(f"adata shape: {adata.shape}")
    print(f"Domains in '{CLUSTER_KEY}': "
          f"{sorted(adata.obs[CLUSTER_KEY].unique().tolist())}")

    # ---- Task 1: web-platform export (domains 3 and 4) ----
    if not args.skip_webtool:
        print("\n" + "=" * 60)
        print(f"Task 1: export web-platform inputs "
              f"({WEBTOOL_GROUPS[0]} and {WEBTOOL_GROUPS[1]})")
        print("=" * 60)
        export_webtool_inputs(adata, WEBTOOL_GROUPS, args.outdir)

    # ---- Task 2: DEG test (3 vs 4) ----
    print("\n" + "=" * 60)
    print(f"Task 2: differential expression ({args.case} vs {args.ref})")
    print("=" * 60)
    de_df, de_signif = run_deg(adata, args.case, args.ref,
                               args.logfc_thresh, args.pval_thresh)
    deg_all_path = _os.path.join(
        args.outdir, f"DEG_{args.case}vs{args.ref}_all_genes.csv")
    deg_sig_path = _os.path.join(
        args.outdir, f"DEG_{args.case}vs{args.ref}_significant.csv")
    de_df.to_csv(deg_all_path, index=False)
    de_signif.to_csv(deg_sig_path, index=False)
    print(f"  Full DEG table: {deg_all_path} ({len(de_df)} genes)")
    print(f"  Significant DEGs: {deg_sig_path} ({len(de_signif)} genes)")
    print("  Top 10 significant DEGs:")
    print(de_signif[["names", "logfoldchanges", "pvals"]].head(10).to_string())

    # ---- Task 3: top-10 heatmap (no row-group legend in the PDAC notebook) ----
    print("\n" + "=" * 60)
    print("Task 3: top-10 DEG heatmap")
    print("=" * 60)
    plot_top10_heatmap(
        adata, de_signif, args.case, args.ref,
        _os.path.join(args.outdir,
                      f"heatmap_top10_DEGs_{args.case}vs{args.ref}.pdf"),
        show_legend=False,
    )

    # ---- Task 4: volcano plot ----
    print("\n" + "=" * 60)
    print("Task 4: volcano plot")
    print("=" * 60)
    plot_volcano(
        de_df, args.case, args.ref, args.logfc_thresh, args.pval_thresh,
        _os.path.join(args.outdir, f"volcano_{args.case}vs{args.ref}.pdf"),
        annotate_fontsize=8, with_title=False,
    )

    # ---- Task 5: GO enrichment (annotation cache shared with the BRCA script)
    if not args.skip_go and len(de_signif) > 0:
        print("\n" + "=" * 60)
        print("Task 5: GO enrichment (hypergeometric test)")
        print("=" * 60)
        prefix = f"go_enrichment_{args.case}vs{args.ref}"
        shared_cache = _os.path.join(
            _os.path.dirname(args.outdir.rstrip("/")), "hsa_go_mapping.csv")
        try:
            go_results = run_go_enrichment(
                de_signif["names"].tolist(), args.outdir, prefix,
                cache_path=shared_cache)
            if go_results is not None and not go_results.empty:
                plot_go_bubble(
                    go_results,
                    _os.path.join(args.outdir,
                                  f"go_bubble_{args.case}vs{args.ref}.pdf"),
                    args.case, args.ref,
                )
        except Exception as e:
            print(f"WARNING: GO enrichment failed ({e}); "
                  "volcano/heatmap results are unaffected.")
    elif args.skip_go:
        print("\nTask 5: GO enrichment skipped (--skip-go).")

    print("\n" + "=" * 60)
    print(f"All PDAC downstream tasks finished. Outputs in: {args.outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
