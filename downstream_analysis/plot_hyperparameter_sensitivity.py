# Auto-detect repo root for reproducibility
import os as _os, sys as _sys
_CURRENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_CURRENT_DIR)
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = _os.path.join(REPO_ROOT, 'saved_results')
_sys.path.insert(0, _os.path.join(REPO_ROOT, 'dcpbst_package'))
import matplotlib.pyplot as plt
import numpy as np
a_values = [0.05, 0.1, 0.2, 0.3, 0.4]  # c
ari_scores = [0.811, 0.833, 0.815, 0.831, 0.808]
nmi_scores = [0.832, 0.832, 0.821, 0.830, 0.819]
x = np.arange(len(a_values))
width = 0.35
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, ari_scores, width, 
                label='ARI', color="#EEAB84", alpha=0.8, edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, nmi_scores, width, 
                label='NMI', color="#cda2f0", alpha=0.8, edgecolor='black', linewidth=0.8)
ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=10)
ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=10)
ax.set_xlabel('Different hyperparameter values of w', fontsize=13)
ax.set_ylabel('Clustering Metrics', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(a_values, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0.7, 0.9)  # y
ax.grid(axis='y', alpha=0.3, linestyle='--')
# 300dpi
plt.tight_layout()
plt.savefig('hyperparameter_c_ari_nmi.png', dpi=300, bbox_inches='tight')
plt.show()