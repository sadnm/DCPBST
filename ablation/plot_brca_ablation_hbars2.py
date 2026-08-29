# Auto-detect repo root for reproducibility
import os as _os, sys as _sys
_CURRENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_CURRENT_DIR)
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = _os.path.join(REPO_ROOT, 'saved_results')
_sys.path.insert(0, _os.path.join(REPO_ROOT, 'dcpbst_package'))
_ABLATION_DIR = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ABLATION_DIR)
"""
Single-seed ablation: 4 HORIZONTAL BAR CHARTS (not boxplots, not rect+whisker).
One metric per subplot: ARI / NMI / Purity / V_Measure (2x2 grid).
  - y-axis:  7 model names (Full on top)
  - x-axis:  metric value  (numbers printed below in axes ticks & in bar tips)
  - each bar tip has value label
Also exports 4 individual PNG/PDF in case user wants separate figures.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DEFAULT_SUMMARY = os.path.join(REPO_ROOT, "BRCA_ablation/BRCA_7ablation_e500_notebook_default/BRCA_ablation_summary.csv")
DEFAULT_OUTDIR  = os.path.join(REPO_ROOT, "BRCA_ablation/BRCA_7ablation_e500_notebook_default")
# 1:1 palette & order matching the user's reference figure, but SOFTER/lighter hues.
# (alpha blending with white in RGB -> 0.55 tint)
REF_COLORS = {
    "DCPBST":                  "#6FA8DC",   # soft blue   (orig #1f77b4 × 0.55 + white)
    "w/o Cross_Attention":     "#F7B26C",   # soft orange (orig #ff7f0e)
    "w/o Imbalance_Regulation":"#7EC97F",   # soft green  (orig #2ca02c)
    "w/o L_DGI":               "#E27F81",   # soft red    (orig #d62728)
    "w/o L_InfoNCE":           "#B294D3",   # soft purple (orig #9467bd)
    "w/o Redundancy_Reduction":"#B58C84",   # soft brown  (orig #8c564b)
    "w/o Spatial_Topology":    "#EFB2DA",   # soft pink   (orig #e377c2)
}
Y_ORDER = [
    "DCPBST",
    "w/o Cross_Attention",
    "w/o Imbalance_Regulation",
    "w/o L_DGI",
    "w/o L_InfoNCE",
    "w/o Redundancy_Reduction",
    "w/o Spatial_Topology",
]
def rename(name):
    s = str(name).strip()
    if s.startswith("Full"): return "DCPBST"
    return s
def plot_one_metric(vals, names, title, xlabel, colors,
                    out_png=None, out_pdf=None, figsize=(7.5, 4.8), dpi=180):
    order = [Y_ORDER.index(n) for n in names]
    idx = np.argsort(order)
    vals = np.asarray(vals)[idx]
    names = np.asarray(names)[idx]
    ys = np.arange(len(names))[::-1]   # top = first name
    bar_colors = [colors[n] for n in names]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.barh(ys, vals, color=bar_colors, edgecolor="#333333",
                   linewidth=0.55, height=0.62, alpha=0.92)
    for y, v, n in zip(ys, vals, names):
        ax.text(v + max(1e-4, (vals.max()-vals.min()))*0.008, y, f"{v:.3f}",
                va="center", ha="left", fontsize=9.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=8)
    xmin, xmax = vals.min(), vals.max()
    span = max(1e-6, xmax - xmin)
    ax.set_xlim(max(0, xmin - 0.18*span), xmax + 0.20*span)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.tick_params(axis="both", labelsize=9.5)
    fig.tight_layout()
    saved = []
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
        saved.append(out_png)
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        saved.append(out_pdf)
    plt.close(fig)
    return saved
def plot(summary_csv=DEFAULT_SUMMARY, outdir=DEFAULT_OUTDIR):
    df = pd.read_csv(summary_csv)
    df["methods"] = df["methods"].apply(rename)
    os.makedirs(outdir, exist_ok=True)
    # Order rule:  DCPBST (Full) is ALWAYS the very TOP on y-axis.
    # The rest 6 ablation variants are sorted by contribution magnitude:
    #   contribution ≈ how much ARI drops when you remove the module = ΔARI%.
    # So the biggest-damage module (contribution largest) comes RIGHT BELOW Full.
    full_row = df[df.methods == "DCPBST"].iloc[0]
    abl_df = df[df.methods != "DCPBST"].copy()
    # Use ΔARI%_median if present; else compute from Full ARI_median
    if "ΔARI%_median" in abl_df.columns:
        abl_df["_contrib"] = abl_df["ΔARI%_median"].astype(float)
    else:
        abl_df["_contrib"] = (
            (float(full_row.ARI_median) - abl_df["ARI_median"].astype(float))
            / float(full_row.ARI_median) * 100.0
        )
    abl_df = abl_df.sort_values("_contrib", ascending=False).reset_index(drop=True)
    ordered_df = pd.concat([pd.DataFrame([full_row]), abl_df],
                           ignore_index=True)
    metrics = [
        ("ARI",       "ARI_median",       "ARI"),
        ("NMI",       "NMI_median",       "NMI"),
        ("Purity",    "Purity_median",    "Purity"),
        ("V_Measure", "V_Measure_median", "V_Measure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=180)
    axes = axes.flatten()
    for ax, (title, val_col, xlab) in zip(axes, metrics):
        vals = ordered_df[val_col].to_numpy(dtype=float)
        names = ordered_df["methods"].to_numpy()
        ys = np.arange(len(ordered_df))[::-1]
        colors = [REF_COLORS[n] for n in names]
        bars = ax.barh(ys, vals, color=colors, edgecolor="#333333",
                       linewidth=0.55, height=0.60, alpha=0.92)
        span = max(1e-6, vals.max() - vals.min())
        for y, v in zip(ys, vals):
            ax.text(v + span*0.012, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=9)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_title(title, fontsize=14, pad=8)
        xmin, xmax = vals.min(), vals.max()
        ax.set_xlim(max(0, xmin - 0.22*span), xmax + 0.23*span)
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    out_png = os.path.join(outdir, "BRCA_ablation_single_seed100_4hbar_2x2.png")
    out_pdf = os.path.join(outdir, "BRCA_ablation_single_seed100_4hbar_2x2.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved combined:", out_png, os.path.getsize(out_png), "B")
    print("Saved combined:", out_pdf, os.path.getsize(out_pdf), "B")
    # 2) Four individual figures (one metric per file)
    # Same top-to-bottom order: Full on top, then 6 ablations by ΔARI% descending.
    per = []
    for title, val_col, xlab in metrics:
        vals = ordered_df[val_col].to_numpy(dtype=float)
        names = ordered_df["methods"].to_numpy()
        safe = xlab.replace("/","_")
        png = os.path.join(outdir, f"BRCA_ablation_single_seed100_hbar_{safe}.png")
        pdf = os.path.join(outdir, f"BRCA_ablation_single_seed100_hbar_{safe}.pdf")
        out = plot_one_metric(vals, names,
                              title=title, xlabel=xlab, colors=REF_COLORS,
                              out_png=png, out_pdf=pdf, figsize=(7.5, 4.8))
        per.extend(out)
    for p in per:
        print("Saved individual:", p, os.path.getsize(p), "B")
    print("\nFinal y-axis order (top → bottom):")
    for i, nm in enumerate(ordered_df["methods"].tolist()):
        c = "Full baseline" if nm == "DCPBST" else \
            (f"ΔARI%={float(abl_df[abl_df.methods==nm]._contrib.iloc[0]):.2f} vs Full")
        print(f"  {i+1}. {nm:<28s}  ({c})")
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default=DEFAULT_SUMMARY)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    plot(args.summary_csv, args.outdir)