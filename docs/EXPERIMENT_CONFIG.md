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

### Unified Model Interface

The **unified DCPBST model** (`dcpbst_package/model.py`) accepts a config dictionary/YAML and supports all dataset variants:

```python
from dcpbst_package.model import Dcpbst

# Load config and instantiate
model = Dcpbst(
    features=[rna_features, img_features_or_coords],  # modality-specific inputs
    adata=adata,
    config=config_dict,  # loaded from configs/*.yaml
    device='cuda',
    n_clusters=n_clusters,
)

# Train
embedding = model.fit(
    epochs=900,
    lr=1e-3,
)

# Cluster
clusters = model.cluster(n_clusters=n_clusters)
```

### Legacy Model Files

Legacy model variants are preserved for exact backward compatibility:

| File | Use Case | Location |
|------|----------|----------|
| `model_827.py` | DLPFC, BRAC, Mouse AREI (Visium) | `dcpbst_package/model_827.py` |
| `model_827_copy.py` | PDAC (Visium, different hyperparameters + POT) | `dcpbst_package/model_827_copy.py` |
| `model_829_copy.py` | Mouse Hypothalamus (MERFISH, no image features) | `dcpbst_package/model_829_copy.py` |
| *(archived)* | Read-only reference copies | `legacy_models/` |

> **Note**: Legacy files are identical in content to their counterparts in `dcpbst_package/`. New experiments should use the unified model.

---

## 1. DLPFC (10x Visium)

**Dataset**: 12 adult human DLPFC (dorsolateral prefrontal cortex) sections, spatial domain identification (cortical layer mapping).  
**Modality**: 10x Visium (RNA + H&E image features)  
**Config file**: `configs/dlpfc_config.yaml`  
**Legacy model**: `dcpbst_package/model_827.py`  
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

### 1.3 DLPFC 12-Section Benchmark

Independent benchmark script: `downstream_analysis/dlpfc_12sections_benchmark.py`

```bash
# Run all 12 sections
python downstream_analysis/dlpfc_12sections_benchmark.py \
  --data_root data \
  --output_dir ./dlpfc_12_results \
  --method mclust \
  --radius 50 \
  --neighbors 7

# Run a subset of sections
python downstream_analysis/dlpfc_12sections_benchmark.py \
  --sections 151507 151508 151673
```

Outputs:
- `DLPFC_12sections_per_section_metrics.csv` — per-section ARI/NMI/Purity/Homogeneity/Completeness/V_Measure
- `DLPFC_12sections_summary_mean_std.csv` — 12-section mean ± std

> **Note**: This benchmark script was derived by extracting the complete pipeline (load → preprocess → image features → DCPBST → clustering → refine → metrics) from the two example notebooks and extending it to loop over all 12 sections. All parameters are aligned with the example notebooks.

### DLPFC Hyperparameters (from `configs/dlpfc_config.yaml`)

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

## 2. BRAC / Human Breast Cancer (10x Visium)

**Dataset**: Human breast cancer tissue, tumor region identification.  
**Modality**: 10x Visium (RNA + H&E image features)  
**Config file**: `configs/brac_config.yaml`  
**Legacy model**: `dcpbst_package/model_827.py`  
**Notebook**: `notebooks/dcpbst_BRAC.ipynb`  
**Downstream notebooks**: `notebooks/dcpbst_BRAC_GO_enrichment_and_violin.ipynb`

| Property | Value |
|----------|-------|
| Data path | `data/Human_breast` |
| n_clusters | Auto-detected from Ground Truth (`7` domains) |
| label_col | `fine_annot_type` (in `metadata.tsv`) |
| Patch data | `data/cut_img_BRCA/cut_img_BRCA.npy`, `Img_feat_brca.npy` |
| DEG contrast groups | Group `8` vs Group `12` (basal-like vs luminal-like tumors, 479 spots) |
| Pre-rendered figures | `figures/BRAC_downanalysis/` |
| Website inputs | `downstream_analysis/website_inputs/` |

Data structure follows 10x Visium format with Ground Truth labels in `Human_breast/metadata.tsv`.

### BRAC Hyperparameters (from `configs/brac_config.yaml`)

Identical architecture to DLPFC:

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

# DEG
deg_group_1: "8"
deg_group_2: "12"

# Misc
seed: 100
use_ot: false
label_col_name: "Ground Truth"
```

---

## 3. PDAC / Pancreatic Cancer (10x Visium)

**Dataset**: Pancreatic ductal adenocarcinoma (PDAC), patient A, ST1 section.  
**Modality**: 10x Visium (RNA + H&E image features)  
**Config file**: `configs/pdac_config.yaml`  
**Legacy model**: `dcpbst_package/model_827_copy.py`  
**Notebook**: `notebooks/dcpbst_PDAC.ipynb`

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

> **Note**: The original notebook used `/data/shijie/dcpbst` as working directory. All paths are now relative to the repo root (`reproducible_experiments/`).

### PDAC-Specific Differences vs. DLPFC/BRAC

| Parameter | DLPFC/BRAC | PDAC |
|-----------|------------|------|
| `pca_n_components` | 1000 | **256** |
| `embed_dim` | 512 | **256** |
| `num_heads` | 8 | **16** |
| `w_kl` | 0.1 | **0.5** |
| `w_pro` | 2.0 | **3.0** |
| `diagnose_every_n_epochs` | 20 | **30** |
| `use_ot` | false | **true** |
| `spatial_s_requires_grad` | false | **true** |

### PDAC Hyperparameters (from `configs/pdac_config.yaml`)

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
**Config file**: `configs/hypothalamus_config.yaml`  
**Legacy model**: `dcpbst_package/model_829_copy.py`  
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

### Hypothalamus Hyperparameters (from `configs/hypothalamus_config.yaml`)

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
**Config file**: `configs/mouse_arei_config.yaml`  
**Legacy model**: `dcpbst_package/model_827.py`  
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

### Mouse AREI Hyperparameters (from `configs/mouse_arei_config.yaml`)

Identical to DLPFC/BRAC architecture:

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

DEG analysis code was extracted from the PDAC notebook (`tutorial_osfs_pdac_10 copy.ipynb`, Cells 2/3/5) into a standalone script:

```bash
python downstream_analysis/differential_gene_analysis.py \
  <path_to_clustered_adata_with_domain_column.h5ad>
```

### Pipeline Details

| Step | Method | Description |
|------|--------|-------------|
| DEG detection | `sc.tl.rank_genes_groups` (Wilcoxon) | Benjamini-Hochberg correction |
| Compatibility | Custom `get_rank_genes_groups_df_compat` | Handles both `dict` and `recarray` scanpy output formats |
| Thresholds | `|log2FC| >= 2` and `adj p-value < 0.05` | Standard DEG filtering |
| Pairwise test | Mann-Whitney U | Cross-cluster pairwise comparison of top 200 genes |
| Outputs | DEG CSVs, volcano plots, violin plots | Per-cluster results |

### Output Files

```
deg_results/
├── DEG_cluster_{id}_all.csv
├── DEG_cluster_{id}_significant.csv
├── DEG_all_clusters_all_genes.csv
├── pairwise_MWU_DEGs_top200genes.csv
├── rank_genes_groups_top25.png
└── rank_genes_groups_violin.png
```

---

## 7. Web Platform Visualization

GO/KEGG enrichment, volcano plots, bubble plots, and differential heatmaps are rendered through a public bioinformatics visualization platform:

🔗 **Platform URL**: [https://cute-companion-liart.vercel.app](https://cute-companion-liart.vercel.app)

### 7.1 BRAC Pre-Packaged Inputs

All BRAC raw input files are pre-packaged in `downstream_analysis/website_inputs/`:

| File | Dimensions | Purpose |
|------|------------|---------|
| `BRAC_heatmap_all_genes_3000.txt` | 3001 × 480 | Full HVG heatmap (Expression Matrix) |
| `BRAC_heatmap_filtered_genes_3000.txt` | 3001 × 480 | Filtered/reordered HVG heatmap |
| `BRAC_heatmap_significant_genes_299.txt` | 300 × 480 | Significant DEG heatmap + GO background genes |
| `BRAC_heatmap_top100_genes.txt` | 101 × 480 | Top-100 DEG preview heatmap |
| `BRAC_sample_class_479.txt` | 480 × 2 | Sample annotations (Group 8 / Group 12) |

### 7.2 BRAC Pre-Rendered Figures

Located at `figures/BRAC_downanalysis/`:

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

PDAC GO/KEGG files and pre-rendered figures are **not available** (lost during archival). To reproduce:

```bash
# Step 1: Run PDAC notebook
jupyter nbconvert --execute --to html notebooks/dcpbst_PDAC.ipynb
# → produces clustered h5ad

# Step 2: Generate DEG CSV
python downstream_analysis/differential_gene_analysis.py ./saved_results1/dcpbst_adata_with_clusters.h5ad
# → generates deg_results/DEG_cluster_<id>_significant.csv

# Step 3: Upload to web platform
# - Volcano plot: CSV with logfoldchanges vs -log10(pvals_adj)
# - GO enrichment: gene symbols from significant DEGs
# - Heatmap: top-299 DEG × sample matrix + group annotations
```

### 7.4 BRAC vs PDAC Notebook Note

The file `notebooks/dcpbst_BRAC_GO_enrichment_and_violin.ipynb` was sourced from `tutorial_dlpfc_brca_go_svg.ipynb`. A separate file `tutorial_dlpfc_pdac_go_svg.ipynb` exists but internally loads `data/Human_breast` (BRAC data) — the "pdac" naming is a historical artifact. Use `differential_gene_analysis.py` for PDAC DEG export instead.

---

## 8. Path Configuration

All paths in the repository are **relative to the repo root** (`reproducible_experiments/`). The setup cell in each notebook auto-detects the repository root.

### Path Mapping from Original Notebooks

| Notebook | Original `os.chdir()` | Reproduction |
|----------|----------------------|--------------|
| DLPFC 151509 / 151671 / BRAC | `/data/dcpbst` | No change needed |
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
│   ├── Human_breast/             # BRAC (10x Visium)
│   ├── PDAC3036911/              # PDAC (h5ad + label CSV + HE image)
│   ├── mouse_Hypothalamus/       # MERFISH .h5ad files
│   ├── Mouse_Brain_Anterior/     # Mouse AREI / Brain Anterior (10x Visium)
│   ├── cut_img_DLPFC/            # Pre-computed DLPFC image patches/features
│   └── cut_img_BRCA/             # Pre-computed BRAC/AREI image patches/features
├── configs/                      # Per-dataset YAML configurations
├── dcpbst_package/                 # Core DCPBST package
├── notebooks/                    # Jupyter notebooks
├── downstream_analysis/          # Standalone scripts
├── figures/                      # Pre-rendered figures
└── legacy_models/                # Archived model files
```

---

## 9. Per-Dataset Config Files Reference

### Config File Summary

| Config File | Dataset | Modality | Model Variant (Legacy) | Key Differences from Default |
|-------------|---------|----------|----------------------|------------------------------|
| `configs/dlpfc_config.yaml` | DLPFC (10x Visium) | RNA + H&E | `model_827.py` | Baseline configuration |
| `configs/brac_config.yaml` | BRAC (10x Visium) | RNA + H&E | `model_827.py` | Same as DLPFC + DEG groups 8 vs 12 |
| `configs/pdac_config.yaml` | PDAC (10x Visium) | RNA + H&E | `model_827_copy.py` | `embed_dim=256`, `num_heads=16`, `use_ot=true`, higher `w_kl`/`w_pro` |
| `configs/hypothalamus_config.yaml` | Mouse Hypothalamus (MERFISH) | RNA + coords | `model_829_copy.py` | `use_img_features=false`, `neighbors=6`, `pca_n_components=256` |
| `configs/mouse_arei_config.yaml` | Mouse AREI (10x Visium) | RNA + H&E | `model_827.py` | Same as DLPFC |

### Architecture Parameters by Dataset

| Parameter | DLPFC / BRAC / AREI | PDAC | Hypothalamus |
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

| Loss | DLPFC / BRAC / AREI | PDAC | Hypothalamus (MERFISH) |
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

| Notebook | Config File | Data Path | Legacy Model |
|----------|------------|-----------|--------------|
| `notebooks/dcpbst_DLPFC_151509.ipynb` | `configs/dlpfc_config.yaml` | `data/151509` | `model_827.py` |
| `notebooks/dcpbst_DLPFC_151671.ipynb` | `configs/dlpfc_config.yaml` | `data/151671` | `model_827.py` |
| `notebooks/dcpbst_BRAC.ipynb` | `configs/brac_config.yaml` | `data/Human_breast` | `model_827.py` |
| `notebooks/dcpbst_BRAC_GO_enrichment_and_violin.ipynb` | *(downstream)* | `data/Human_breast` | *(post-analysis)* |
| `notebooks/dcpbst_PDAC.ipynb` | `configs/pdac_config.yaml` | `data/PDAC3036911` | `model_827_copy.py` |
| `notebooks/dcpbst_Mouse_Hypothalamus.ipynb` | `configs/hypothalamus_config.yaml` | `data/mouse_Hypothalamus` | `model_829_copy.py` |
| `notebooks/dcpbst_Mouse_AREI.ipynb` | `configs/mouse_arei_config.yaml` | `data/Mouse_Brain_Anterior` | `model_827.py` |

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