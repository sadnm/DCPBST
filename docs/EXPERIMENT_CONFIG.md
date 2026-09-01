# DCPBST Experiment Configuration & Reproduction Record

This document records the model versions, hyperparameters, data paths, and evaluation metrics used for each dataset in the manuscript *"Spatial Domain Identification via Multi-Modal Integration"*. It serves as a comprehensive reference for reproducing all main experimental results.

---

## 0. Common Parameters (All Datasets)

These parameters are shared across all dataset configurations unless explicitly overridden.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Random seed | 100 | Fixes `numpy`, `torch`, `random`, `cudnn` (deterministic mode, benchmark disabled) |
| Clustering method | `mclust` (primary) | R `mclust` with `EEE` model via `rpy2`; fallback: `kmeans`, `leiden` |
| Clustering seed (R) | 2024 | `set.seed(2024)` inside R mclust call |
| Refine radius | 50 | Spatial neighbor label smoothing radius |
| HVG selection | 3000 | `seurat_v3` method for highly variable gene selection |
| Normalization | `target_sum=1e4`, `log1p` | Standard scanpy pipeline |
| PCA (pre-clustering) | 50 components | Applied to embeddings before clustering |
| Device | `cuda` (fallback `cpu`) | GPU acceleration when available |
| Optimizer | Adam | For model training |

### Model Variant Files

Each notebook imports one model variant from `dcpbst_package/` and passes all
hyperparameters **directly in code** (there is no configuration file or YAML
loader in the repository — every dataset is self-contained in its notebook):

| File | Used by | Location |
|------|---------|----------|
| `model_827.py` | DLPFC (151509, 151671), Mouse AREI (Visium) | `dcpbst_package/model_827.py` |
| `model_827_copy.py` | BRCA, PDAC (Visium; PDAC passes different hyperparameters + POT) | `dcpbst_package/model_827_copy.py` |
| `model_829_copy.py` | Mouse Hypothalamus (MERFISH, no image features) | `dcpbst_package/model_829_copy.py` |

```python
from dcpbst_package.model_827_copy import Dcpbst

# Instantiate (hyperparameters passed directly; PDAC overrides them in its notebook)
model = Dcpbst(
    features=[rna_features, img_features_or_coords],  # modality-specific inputs
    adata=adata,
    device='cuda',
    n_clusters=n_clusters,
)

# Train
embedding = model.fit(epochs=900, lr=1e-3, ...)

# Cluster
clusters = model.cluster(n_clusters=n_clusters)
```

---

## 1. DLPFC (10x Visium)

**Dataset**: 12 adult human DLPFC (dorsolateral prefrontal cortex) sections, spatial domain identification (cortical layer mapping).  
**Modality**: 10x Visium (RNA + H&E image features)  
**Model variant**: `dcpbst_package/model_827.py`  
**Notebook**: `notebooks/dcpbst_DLPFC_151509.ipynb` (section 151509), `notebooks/dcpbst_DLPFC_151671.ipynb` (section 151671)

### 1.1 Example Sections (151509 and 151671)

| Property | Section 151509 | Section 151671 |
|-----------|----------------|----------------|
| Notebook file | `notebooks/dcpbst_DLPFC_151509.ipynb` | `notebooks/dcpbst_DLPFC_151671.ipynb` |
| Data path | `data/151509` | `data/151671` |
| n_clusters | **7** (from Ground Truth) | **5** (from Ground Truth) |
| label_col | `layer_guess` | `layer_guess` |
| Patch data | `data/cut_img_DLPFC/cut_img_151509.npy`, `Img_feat_151509.npy` | `data/cut_img_DLPFC/cut_img_151671.npy`, `Img_feat_151671.npy` |
| Metrics reported | ARI, NMI, Purity, V_Measure | ARI, NMI, Purity, V_Measure |

### 1.2 All 12 Sections

Complete 12-section IDs:

```
151507, 151508, 151509, 151510,
151669, 151670, 151671, 151672,
151673, 151674, 151675, 151676
```

Each section follows the 10x Visium directory structure:

```
data/{section_id}/
├── filtered_feature_bc_matrix.h5
├── raw_feature_bc_matrix.h5
├── metadata.tsv            # Ground Truth; column: layer_guess
└── spatial/
    ├── full_image.tif      # H&E large image
    ├── tissue_hires_image.png
    ├── tissue_lowres_image.png
    ├── scalefactors_json.json
    └── tissue_positions_list.csv
```

Image patch data (pre-computed, optionally pre-loaded):
- Patches: `data/cut_img_DLPFC/cut_img_{section_id}.npy`
- ResNet features: `data/cut_img_DLPFC/Img_feat_{section_id}.npy`

### DLPFC Hyperparameters

All hyperparameters are passed directly in the notebook (no configuration
file is read):

```yaml
# Model architecture
latent_dim: 1024
pca_n_components: 1000
embed_dim: 512
num_heads: 8
output_dim: 256

# Graph construction
neighbors: 7
gat_dropout: 0.5

# Training
epochs: 900
lr: 0.001
w_cls: 10.0
w_recon: 10.0
w_kl: 0.1
w_pro: 2.0
w_info: 3.0
w_dgi: 0.1
w_clu: 1.0
diagnose_every_n_epochs: 20

# Clustering
cluster_method: mclust
refine_radius: 50
n_clusters: 7  # auto-detected from Ground Truth; 7 for 151509, 5 for 151671

# Misc
seed: 100
use_ot: false
label_col_name: "Ground Truth"
```

---

## 2. BRCA / Human Breast Cancer (10x Visium)

**Dataset**: Human breast cancer tissue, tumor region identification.  
**Modality**: 10x Visium (RNA + H&E image features)  
**Model variant**: `dcpbst_package/model_827_copy.py`  
**Notebook**: `notebooks/dcpbst_BRAC.ipynb` (filename kept as-is)  
**Downstream script**: `downstream_analysis/run_brca_deg.py` (web-platform export / DEG / heatmap / volcano / GO; shared helpers in `deg_common.py`)

| Property | Value |
|----------|-------|
| Data path | `data/Human_breast` |
| n_clusters | Auto-detected from Ground Truth |
| label_col | `fine_annot_type` (in `metadata.tsv`) |
| Patch data | `data/cut_img_BRCA/cut_img_BRCA.npy`, `Img_feat_brca.npy` |
| DEG contrast groups | Cluster `12` vs cluster `14` (121 significant DEGs, Wilcoxon, \|log2FC\| ≥ 2, P < 0.05) |
| Web-platform export | Domains `12` vs `14` (626 spots: 431 + 195), generated by `run_brca_deg.py` into `downstream_analysis/deg_results/brca/webtool_*.txt` |
| Pre-rendered figures | `figures/BRCA_downanalysis/` |

Data structure follows 10x Visium format with Ground Truth labels in `Human_breast/metadata.tsv`.

### BRCA Hyperparameters

Identical architecture to DLPFC; all hyperparameters are passed directly in
the notebook (no configuration file is read):

```yaml
# Model architecture
latent_dim: 1024
pca_n_components: 1000
embed_dim: 512
num_heads: 8
output_dim: 256

# Graph construction
neighbors: 7
gat_dropout: 0.5

# Training
epochs: 900
lr: 0.001
w_cls: 10.0
w_recon: 10.0
w_kl: 0.1
w_pro: 2.0
w_info: 3.0
w_dgi: 0.1
w_clu: 1.0
diagnose_every_n_epochs: 20

# Clustering
cluster_method: mclust
refine_radius: 50
n_clusters: 7  # auto-detected

# DEG / web-platform export (downstream script run_brca_deg.py)
case_group: "12"
ref_group: "14"
webtool_groups: ["12", "14"]

# Misc
seed: 100
use_ot: false
label_col_name: "Ground Truth"
```

---

## 3. PDAC / Pancreatic Cancer (10x Visium)

**Dataset**: Pancreatic ductal adenocarcinoma (PDAC), patient A, ST1 section.  
**Modality**: 10x Visium (RNA + H&E image features)  
**Model variant**: `dcpbst_package/model_827_copy.py` (PDAC-specific hyperparameters are passed directly in the notebook)  
**Notebook**: `notebooks/dcpbst_PDAC.ipynb`  
**Downstream script**: `downstream_analysis/run_pdac_deg.py` (web-platform export / DEG / heatmap / volcano / GO; shared helpers in `deg_common.py`)

| Property | Value |
|----------|-------|
| Data path | `data/PDAC3036911` |
| h5ad file | `data/PDAC3036911/sorted_GSM3036911.h5ad` |
| Label file | `data/PDAC3036911/20210201_PatientA1_label_chinese.csv` |
| H&E image | `data/PDAC3036911/GSM3036911_PDAC-A-ST1-HE.jpg` |
| n_clusters | Auto-detected from `Ground Truth` (≈ 7) |
| label_col | `ground_truth` (loaded from label CSV) |
| use_ot | `true` — PDAC uses optimal transport alignment |
| spatial_s_requires_grad | `true` |
| DEG contrast groups | Cluster `3` vs cluster `4` (66 significant DEGs, Wilcoxon, \|log2FC\| ≥ 2, P < 0.05) |
| Web-platform export | Domains `3` and `4` (235 spots: 66 + 169), generated by `run_pdac_deg.py` into `downstream_analysis/deg_results/pdac/webtool_*.txt` |

> **Note**: The original notebook used `/data/shijie/dcpbst` as working directory. All paths are now relative to the repo root (`reproducible_experiments/`).

### PDAC-Specific Differences vs. DLPFC/BRCA

| Parameter | DLPFC/BRCA | PDAC |
|-----------|------------|------|
| `pca_n_components` | 1000 | **256** |
| `embed_dim` | 512 | **256** |
| `num_heads` | 8 | **16** |
| `w_kl` | 0.1 | **0.5** |
| `w_pro` | 2.0 | **3.0** |
| `diagnose_every_n_epochs` | 20 | **30** |
| `use_ot` | false | **true** |
| `spatial_s_requires_grad` | false | **true** |

### PDAC Hyperparameters

PDAC-specific values passed directly in the notebook (no configuration file
is read):

```yaml
# Model architecture (PDAC-specific)
latent_dim: 1024
pca_n_components: 256
embed_dim: 256
num_heads: 16
output_dim: 256

# Graph construction
neighbors: 7
gat_dropout: 0.5

# Training (PDAC-specific loss weights)
epochs: 900
lr: 0.001
w_cls: 10.0
w_recon: 10.0
w_kl: 0.5
w_pro: 3.0
w_info: 3.0
w_dgi: 0.1
w_clu: 1.0
diagnose_every_n_epochs: 30

# Clustering
cluster_method: mclust
refine_radius: 50
n_clusters: 7  # auto-detected

# Misc
seed: 100
use_ot: true
spatial_s_requires_grad: true
label_col_name: "Ground Truth"
```

---

## 4. Mouse Hypothalamus (MERFISH)

**Dataset**: Mouse hypothalamus, neuronal subtype identification.  
**Modality**: MERFISH (RNA + spatial coordinates, **no image features**)  
**Model variant**: `dcpbst_package/model_829_copy.py`  
**Notebook**: `notebooks/dcpbst_Mouse_Hypothalamus.ipynb`

| Property | Value |
|----------|-------|
| Data file | `data/mouse_Hypothalamus/MERFISH_0.14_20251020183913_1.h5ad` |
| Data directory | `data/mouse_Hypothalamus` |
| Backup h5ad | `MERFISH_0.24_20251119191016.h5ad` |
| n_clusters | Auto-detected from `ground_truth` column |
| label_col | `ground_truth` (in h5ad `obs`) |
| use_img_features | `false` — MERFISH uses spatial coordinates only |
| neighbors | **6** (vs. 7 for Visium) |
| label_col_name | `"ground_truth"` (lowercase) |

> **Note**: The MERFISH data was originally located at `/data/shijie/dcpbst/data/mouse_Hypothalamus/`. It was copied to the repo's `data/` directory and symlinked back.

### Hypothalamus Hyperparameters

MERFISH-specific values passed directly in the notebook (no configuration
file is read):

```yaml
# Model architecture (MERFISH-specific)
latent_dim: 1024
pca_n_components: 256
embed_dim: 512
num_heads: 8
output_dim: 256

# Graph construction (MERFISH: fewer neighbors)
neighbors: 6
gat_dropout: 0.5

# Training
epochs: 900
lr: 0.001
w_cls: 10.0
w_recon: 10.0
w_kl: 0.1
w_pro: 2.0
w_info: 3.0
w_dgi: 0.1
w_clu: 1.0
diagnose_every_n_epochs: 20

# Clustering
cluster_method: mclust
refine_radius: 50
n_clusters: null  # auto-detected

# Misc
seed: 100
use_ot: false
label_col_name: "ground_truth"
```

---

## 5. Mouse AREI / Brain Anterior (10x Visium)

**Dataset**: Mouse brain anterior region.  
**Modality**: 10x Visium (RNA + H&E image features)  
**Model variant**: `dcpbst_package/model_827.py`  
**Notebook**: `notebooks/dcpbst_Mouse_AREI.ipynb`

| Property | Value |
|----------|-------|
| Data path | `data/Mouse_Brain_Anterior` |
| n_clusters | Auto-detected from `Ground Truth` |
| label_col | `ground_truth` (in `metadata.tsv`) |
| Patch data | `data/cut_img_BRCA/cut_img_MouseAnterior.npy`, `Img_feat_MouseAnterior.npy` |
| Ground Truth file | `data/Mouse_Brain_Anterior/truth.txt` |
| Embeddings file | `data/Mouse_Brain_Anterior/embeddings.npy` |

Data structure follows 10x Visium format with `spatial/` directory and `full_image.tif`.

### Mouse AREI Hyperparameters

Identical to DLPFC/BRCA architecture; values passed directly in the notebook
(no configuration file is read):

```yaml
# Model architecture
latent_dim: 1024
pca_n_components: 1000
embed_dim: 512
num_heads: 8
output_dim: 256

# Graph construction
neighbors: 7
gat_dropout: 0.5

# Training
epochs: 900
lr: 0.001
w_cls: 10.0
w_recon: 10.0
w_kl: 0.1
w_pro: 2.0
w_info: 3.0
w_dgi: 0.1
w_clu: 1.0
diagnose_every_n_epochs: 20

# Clustering
cluster_method: mclust
refine_radius: 50
n_clusters: null  # auto-detected

# Misc
seed: 100
use_ot: false
label_col_name: "Ground Truth"
```

---

## 6. Differential Gene Analysis (Downstream)

The notebooks only demonstrate spatial-domain identification. All DEG-related
tasks were extracted into two standalone driver scripts that share the same
analysis code in `downstream_analysis/deg_common.py` (no GPU required; they
consume the clustered h5ad saved by the notebooks):

```bash
# PDAC: web-platform export (domains 3 and 4) + DEG 3 vs 4
#       + heatmap + volcano + GO enrichment
python downstream_analysis/run_pdac_deg.py

# BRCA: web-platform export (domains 12 vs 14) + DEG 12 vs 14
#       + heatmap + two volcano plots + GO enrichment
python downstream_analysis/run_brca_deg.py

# Offline / fast runs
python downstream_analysis/run_pdac_deg.py --skip-go
python downstream_analysis/run_brca_deg.py --skip-go --skip-webtool
```

### Pipeline Details

| Step | Method | Description |
|------|--------|-------------|
| DEG detection | `sc.tl.rank_genes_groups` (Wilcoxon rank-sum) | Case domain vs reference domain |
| Compatibility | `get_rank_genes_groups_df_compat` (in `deg_common.py`) | Handles both `dict` and `recarray` scanpy output formats |
| Thresholds | `|log2FC| >= 2` and `p-value < 0.05` | Significant DEG filtering |
| Heatmap | seaborn clustermap | Top-10 DEGs (5 up / 5 down), cluster-mean expression, hierarchical gene ordering |
| Volcano | `-log10(P)` vs log2FC | BRCA additionally provides a reference-perspective volcano |
| GO enrichment | Hypergeometric test + BH correction | Human GOA annotation; cached once in `deg_results/hsa_go_mapping.csv` and shared by both scripts |

### Output Files

```
downstream_analysis/deg_results/
├── hsa_go_mapping.csv                 # human GO annotation cache (auto-downloaded, shared)
├── pdac/
│   ├── DEG_3vs4_all_genes.csv         # 3000 genes tested
│   ├── DEG_3vs4_significant.csv       # 66 significant DEGs
│   ├── heatmap_top10_DEGs_3vs4.pdf
│   ├── volcano_3vs4.pdf
│   ├── go_enrichment_3vs4_all_results.csv / _significant.csv  # 229 terms
│   ├── go_bubble_3vs4.pdf
│   ├── webtool_data_heatmap.txt       # raw genes x spots matrix (3000 x 235)
│   └── webtool_sample_class.txt       # spot group labels (domains 3 / 4; 66 + 169)
└── brca/
    ├── DEG_12vs14_all_genes.csv       # 3000 genes tested
    ├── DEG_12vs14_significant.csv     # 121 significant DEGs
    ├── heatmap_top10_DEGs_12vs14.pdf
    ├── volcano_12vs14.pdf
    ├── volcano_14based_12vs14.pdf     # reference-perspective volcano
    ├── go_enrichment_12vs14_all_results.csv / _significant.csv  # 131 terms
    ├── go_bubble_12vs14.pdf
    ├── webtool_data_heatmap.txt       # raw genes x spots matrix (3000 x 626)
    └── webtool_sample_class.txt       # spot group labels (domains 12 / 14; 431 + 195)
```

---

## 7. Web Platform Visualization

GO/KEGG enrichment, volcano plots, bubble plots, and differential heatmaps are rendered through a public bioinformatics visualization platform:

🔗 **Platform URL**: [https://cute-companion-liart.vercel.app](https://cute-companion-liart.vercel.app)

### 7.1 Web-Platform Inputs (BRCA and PDAC)

Each downstream driver exports one raw expression set for the web platform
(Task 1) into `downstream_analysis/deg_results/<dataset>/` — these are the
single raw gene-expression exports kept per dataset:

| Dataset | Generator | Groups | Spots | Files |
|---------|-----------|--------|-------|-------|
| BRCA | `run_brca_deg.py` | domains `12` vs `14` | 626 (431 + 195) | `deg_results/brca/webtool_*.txt` |
| PDAC | `run_pdac_deg.py` | domains `3` and `4` | 235 (66 + 169) | `deg_results/pdac/webtool_*.txt` |

| File | Dimensions | Purpose |
|------|------------|---------|
| `webtool_data_heatmap.txt` | 3000 genes × N spots (tab-separated) | Raw expression matrix (Expression Matrix input) |
| `webtool_sample_class.txt` | N spots × 2 columns | Spot group annotations (domain labels) |

Regenerate at any time with `python downstream_analysis/run_brca_deg.py` or
`run_pdac_deg.py` (skip the other tasks with `--skip-go` if only the export
is needed).

### 7.2 BRCA Pre-Rendered Figures

Located at `figures/BRCA_downanalysis/`:

- `go.bf196e554962720c/` — GO three ontologies (BP, MF, CC)
  - `{BP,MF,CC}_{goplot,Enrichment_Score_barplot,dotplot,emap,cnetplot}.{png,pdf,tiff}`
  - `GO_Three_Ontologies.{png,pdf,tiff}`
  - `go.{BP,MF,CC}.{all,sig}.xlsx`
  - `{BP,MF,CC}.RData`
- `pathway.bf196e554962720c/` — KEGG/Pathway
  - `Pathway_{Enrichment_Score_barplot,dotplot,emap,cnetplot}.{png,pdf,tiff}`
  - `pathway.{sig,all}.xlsx`, `KEGG.RData`
  - `hsa*.pathview.png` (e.g., hsa04148, hsa05217 pathway views)

### 7.3 PDAC Enrichment Regeneration

PDAC DEG/GO results are reproduced end-to-end by the downstream driver script
(no web platform required for GO — the hypergeometric enrichment is built in):

```bash
# Step 1: Run the PDAC notebook (spatial-domain identification)
jupyter notebook notebooks/dcpbst_PDAC.ipynb
# → trains the model and saves saved_results/dcpbst_PDAC_adata_with_clusters.h5ad
#   (plus dcpbst_PDAC_model.pth and dcpbst_PDAC_clustering_metrics.csv)

# Step 2: Run the PDAC downstream pipeline
python downstream_analysis/run_pdac_deg.py
# → deg_results/pdac/ : webtool_*.txt export (domains 3 and 4, 235 spots),
#   DEG tables (66 significant DEGs, 3 vs 4), heatmap + volcano PDFs,
#   GO enrichment (229 terms) + bubble plot

# Step 3 (optional): upload DEG CSVs to the web platform
# - Volcano plot: CSV with logfoldchanges vs -log10(pvals)
# - GO/KEGG: gene symbols from DEG_3vs4_significant.csv
```

### 7.4 Downstream-Code Note

The notebooks deliberately contain **no** DEG/GO cells — they only demonstrate
spatial-domain identification. All downstream tasks (DEG tables, heatmaps,
volcano plots, GO enrichment, web-platform export) live in the two driver
scripts `downstream_analysis/run_brca_deg.py` and
`downstream_analysis/run_pdac_deg.py`, which share their analysis code in
`downstream_analysis/deg_common.py`. Pre-rendered BRCA GO/KEGG figures from
the web platform are archived in `figures/BRCA_downanalysis/`.

---

## 8. Path Configuration

All paths in the repository are **relative to the repo root** (`reproducible_experiments/`). The setup cell in each notebook auto-detects the repository root.

### Path Mapping from Original Notebooks

| Notebook | Original `os.chdir()` | Reproduction |
|----------|----------------------|--------------|
| DLPFC 151509 / 151671 / BRCA | `/data/dcpbst` | No change needed |
| PDAC / xiaqiu / AREI | `/data/shijie/dcpbst` | **Changed to `/data/dcpbst`** |

### Quick Path Fix Script

```bash
cd reproducible_experiments/notebooks
for f in *.ipynb; do
  python3 -c "
import json, sys
nb = json.load(open('$f'))
for cell in nb['cells']:
    if cell['cell_type']=='code':
        cell['source']=[s.replace('/data/shijie/dcpbst','/data/dcpbst') for s in cell['source']]
json.dump(nb, open('$f','w'), indent=1, ensure_ascii=False)
"
done
```

### Data Directory Structure

```
reproducible_experiments/
├── data/
│   ├── 151507 … 151676/          # 12 DLPFC Visium sections
│   ├── Human_breast/             # BRCA (10x Visium)
│   ├── PDAC3036911/              # PDAC (h5ad + label CSV + HE image)
│   ├── mouse_Hypothalamus/       # MERFISH .h5ad files
│   ├── Mouse_Brain_Anterior/     # Mouse AREI / Brain Anterior (10x Visium)
│   ├── cut_img_DLPFC/            # Pre-computed DLPFC image patches/features
│   └── cut_img_BRCA/             # Pre-computed BRCA/AREI image patches/features
├── dcpbst_package/               # Core DCPBST package (all model variants)
├── notebooks/                    # Jupyter notebooks (spatial-domain identification)
├── ablation/                     # Ablation study scripts (BRCA only)
├── downstream_analysis/          # DEG / GO driver scripts + deg_common.py
├── figures/                      # Pre-rendered figures
└── saved_results/                # Notebook outputs (generated locally; not shipped)
```

---

## 9. Per-Dataset Parameter Reference

> All hyperparameters below are passed directly in code (notebook cells /
> model variant defaults). The repository contains **no configuration files
> or YAML loader** — each notebook is self-contained.

### Model Variant Summary

| Dataset | Modality | Model Variant | Key Differences |
|---------|----------|---------------|-----------------|
| DLPFC (10x Visium) | RNA + H&E | `model_827.py` | Baseline configuration |
| BRCA (10x Visium) | RNA + H&E | `model_827_copy.py` | Same architecture as DLPFC; web export domains 12 vs 14, DEG 12 vs 14 |
| PDAC (10x Visium) | RNA + H&E | `model_827_copy.py` | `embed_dim=256`, `num_heads=16`, `use_ot=true`, higher `w_kl`/`w_pro` |
| Mouse Hypothalamus (MERFISH) | RNA + coords | `model_829_copy.py` | `use_img_features=false`, `neighbors=6`, `pca_n_components=256` |
| Mouse AREI (10x Visium) | RNA + H&E | `model_827.py` | Same as DLPFC |

### Architecture Parameters by Dataset

| Parameter | DLPFC / BRCA / AREI | PDAC | Hypothalamus |
|-----------|--------------------|------|--------------|
| `latent_dim` | 1024 | 1024 | 1024 |
| `pca_n_components` | 1000 | 256 | 256 |
| `embed_dim` | 512 | 256 | 512 |
| `num_heads` | 8 | 16 | 8 |
| `output_dim` | 256 | 256 | 256 |
| `neighbors` | 7 | 7 | 6 |
| `use_img_features` | true | true | false |
| `use_ot` | false | true | false |
| `spatial_s_requires_grad` | false | true | false |
| `diagnose_every_n_epochs` | 20 | 30 | 20 |

### Loss Weights by Dataset

| Loss | DLPFC / BRCA / AREI | PDAC | Hypothalamus (MERFISH) |
|------|--------------------|------|------------------------|
| `w_cls` | 10.0 | 10.0 | 10.0 |
| `w_recon` | 10.0 | 10.0 | 10.0 |
| `w_kl` | 0.1 | **0.5** | 0.1 |
| `w_pro` | 2.0 | **3.0** | 2.0 |
| `w_info` | 3.0 | 3.0 | 3.0 |
| `w_dgi` | 0.1 | 0.1 | 0.1 |
| `w_clu` | 1.0 | 1.0 | 1.0 |

All datasets use `epochs=900`, `lr=0.001`, `seed=100`.

### Notebook-to-Dataset Mapping

| Notebook | Data Path | Model Variant |
|----------|-----------|---------------|
| `notebooks/dcpbst_DLPFC_151509.ipynb` | `data/151509` | `model_827.py` |
| `notebooks/dcpbst_DLPFC_151671.ipynb` | `data/151671` | `model_827.py` |
| `notebooks/dcpbst_BRAC.ipynb` (filename kept as-is) | `data/Human_breast` | `model_827_copy.py` |
| `notebooks/dcpbst_PDAC.ipynb` | `data/PDAC3036911` | `model_827_copy.py` |
| `notebooks/dcpbst_Mouse_Hypothalamus.ipynb` | `data/mouse_Hypothalamus` | `model_829_copy.py` |
| `notebooks/dcpbst_Mouse_AREI.ipynb` | `data/Mouse_Brain_Anterior` | `model_827.py` |

> Downstream (DEG / GO) tasks are not in the notebooks — they are reproduced
> by `downstream_analysis/run_brca_deg.py` and
> `downstream_analysis/run_pdac_deg.py` (shared code in `deg_common.py`).

---

## Appendix: Evaluation Metrics

All experiments report the following standard clustering metrics:

| Metric | Description |
|--------|-------------|
| ARI | Adjusted Rand Index |
| NMI | Normalized Mutual Information |
| Purity | Clustering purity |
| Homogeneity | Homogeneity score |
| Completeness | Completeness score |
| V_Measure | Harmonic mean of homogeneity and completeness |

These are computed via `sklearn.metrics` in `dcpbst_package/evals.py`.