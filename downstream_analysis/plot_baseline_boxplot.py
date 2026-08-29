# Auto-detect repo root for reproducibility
import os as _os, sys as _sys
_CURRENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_CURRENT_DIR)
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = _os.path.join(REPO_ROOT, 'saved_results')
_sys.path.insert(0, _os.path.join(REPO_ROOT, 'dcpbst_package'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
baseline_dir = os.path.join(REPO_ROOT, "DLPFC_data")
output_dir = os.path.join(REPO_ROOT, 'saved_results')
csv_files = sorted(glob.glob(os.path.join(baseline_dir, 'DLPFC_batch_*_results.csv')))
all_data = []
for f in csv_files:
    df = pd.read_csv(f)
    all_data.append(df)
data = pd.concat(all_data, ignore_index=True)
method_order = [
    'DCPBST',
    'DeepST',
    'GraphST',
    'SDUCL',
    'SEDR',
    'SpaGCN',
    'spCLUE',
    'stLearn'
]
method_colors = {
    'DCPBST': '#1f77b4',
    'DeepST': '#ff7f0e',
    'GraphST': '#2ca02c',
    'SDUCL': '#d62728',
    'SEDR': '#9467bd',
    'SpaGCN': '#8c564b',
    'spCLUE': '#e377c2',
    'stLearn': '#7f7f7f'
}
metrics = ['ARI', 'NMI', 'Purity', 'V_Measure']
x_ranges = {
    'ARI': (0.18, 0.95),
    'NMI': (0.38, 0.92),
    'Purity': (0.48, 1.01),
    'V_Measure': (0.38, 0.92)
}
plt.rcParams.update({
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()
for i, metric in enumerate(metrics):
    ax = axes[i]
    plot_data = []
    labels = []
    for method in method_order:
        vals = data[data['methods'] == method][metric].dropna().values
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
        flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5}
    )
    for patch, method in zip(box['boxes'], labels):
        patch.set_facecolor(method_colors[method])
        patch.set_alpha(0.85)
        patch.set_linewidth(1.2)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_title(metric, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=11)
    if metric in x_ranges:
        ax.set_xlim(x_ranges[metric])
    ax.invert_yaxis()
fig.text(0.01, 0.96, 'E', fontsize=28, fontweight='bold', va='top')
plt.tight_layout(rect=[0.03, 0, 1, 0.96])
output_path = os.path.join(output_dir, 'baseline_comparison_boxplot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to {output_path}")
print("\n=== Summary Statistics ===")
for metric in metrics:
    print(f"\n--- {metric} ---")
    for method in method_order:
        vals = data[data['methods'] == method][metric].dropna()
        if len(vals) > 0:
            print(f"{method:12s}: mean={vals.mean():.4f}, median={vals.median():.4f}, std={vals.std():.4f}")