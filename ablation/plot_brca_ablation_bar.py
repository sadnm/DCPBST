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
Single standalone ablation figure (NOT boxplots) for BRCA single-seed (seed=100).
Produces a single grouped bar chart:
  x-axis: 6 ablation modules (ordered by ΔARI% descending = damage magnitude)
  left y-axis: ΔARI% (main metric, solid bars)  right y-axis: ΔNMI% (hatched bars)
  Top 3 core modules (Spatial/Redundancy/L_DGI) highlighted in a distinct palette.
Outputs PNG + PDF to the result directory.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import container
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY = os.path.join(REPO_ROOT, "BRCA_ablation/BRCA_7ablation_e500_notebook_default/BRCA_ablation_summary.csv")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "BRCA_ablation/BRCA_7ablation_e500_notebook_default")
def plot(summary_csv=DEFAULT_SUMMARY, outdir=DEFAULT_OUTDIR):
    df = pd.read_csv(summary_csv)
    # Drop Full (baseline line) -> 6 ablation bars
    abl = df[~df["methods"].str.strip().str.startswith("Full")].copy()
    # Order by damage magnitude (biggest delta first)
    abl = abl.sort_values("ΔARI%_median", ascending=False).reset_index(drop=True)
    labels = abl["methods"].tolist()
    x = np.arange(len(labels))
    dari = abl["ΔARI%_median"].to_numpy(dtype=float)
    dnmi = abl["ΔNMI%_median"].to_numpy(dtype=float)
    # Palette: top 3 core modules get warm/strong colors, rest muted
    core_idx = [0, 1]  # Spatial_Topology and Redundancy_Reduction are always top 2
    for i in range(len(abl)):
        if "L_DGI" in abl.iloc[i]["methods"]:
            core_idx.append(i); break
    colors_ari = []
    for i in range(len(abl)):
        if i == core_idx[0]:
            colors_ari.append("#D62728")   # red: Spatial (largest drop)
        elif i == core_idx[1]:
            colors_ari.append("#FF7F0E")   # orange: Redundancy
        elif i in core_idx:
            colors_ari.append("#2CA02C")   # green: L_DGI (3rd)
        else:
            colors_ari.append("#7f7f7f")   # gray for rest
    colors_nmi = ["#9ECAE1" if c == "#7f7f7f" else c for c in colors_ari]
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(9.2, 5.0), dpi=160)
    ax2 = ax1.twinx()
    bars1 = ax1.bar(x - width/2, dari, width=width, color=colors_ari,
                    edgecolor="black", linewidth=0.6, label="ΔARI% (median vs Full)")
    bars2 = ax2.bar(x + width/2, dnmi, width=width, color=colors_nmi,
                    edgecolor="black", linewidth=0.6, hatch="//", alpha=0.85,
                    label="ΔNMI% (median vs Full)")
    # Value labels on top of each bar
    for b, v in zip(bars1, dari):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.18, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    for b, v in zip(bars2, dnmi):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.08, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=8, color="#1f4e79")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right", fontsize=9.5)
    ax1.set_ylabel("ΔARI% vs Full model  (higher = bigger damage when removed)",
                   fontsize=10.5)
    ax2.set_ylabel("ΔNMI% vs Full model  (higher = bigger damage when removed)",
                   fontsize=10.5, color="#1f4e79")
    ax1.set_ylim(0, max(16, dari.max() * 1.18))
    ax2.set_ylim(0, max(6, dnmi.max() * 1.25))
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    title = ("Ablation Study on BRCA (Human Breast) — single seed = 100  |  "
             "Full model ARI = %.3f, NMI = %.3f" % (
                 float(df[df["methods"].str.startswith("Full")]["ARI_median"].iloc[0]),
                 float(df[df["methods"].str.startswith("Full")]["NMI_median"].iloc[0])
             ))
    ax1.set_title(title, fontsize=11, pad=12)
    # Legend (deduplicate)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Remove the errorbar container duplicates if any
    h1 = [x for x in h1 if not isinstance(x, container.ErrorbarContainer)]
    h2 = [x for x in h2 if not isinstance(x, container.ErrorbarContainer)]
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, framealpha=0.9)
    # Annotate top-3 core components claim
    ax1.annotate(
        "Top-3 core:  Spatial_Topology  >  Redundancy_Reduction  >  L_DGI",
        xy=(0.01, 0.985), xycoords="axes fraction",
        va="top", ha="left", fontsize=9, color="#333",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF3CD", ec="#D6B656", lw=0.7),
    )
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, "BRCA_ablation_single_seed100_bar.png")
    out_pdf = os.path.join(outdir, "BRCA_ablation_single_seed100_bar.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out_png, "(", os.path.getsize(out_png), "bytes )")
    print("Saved:", out_pdf, "(", os.path.getsize(out_pdf), "bytes )")
    print("Methods order (biggest damage → smallest):")
    for i, (lbl, a, n) in enumerate(zip(labels, dari, dnmi)):
        print(f"  {i+1:>2}. {lbl:<28s}   ΔARI%={a:5.2f}   ΔNMI%={n:5.2f}")
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default=DEFAULT_SUMMARY)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    a = ap.parse_args()
    plot(a.summary_csv, a.outdir)