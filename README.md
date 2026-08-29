# DCPBST: Deciphering Spatial Domains from Spatial Transcriptomics via Dual Graph Contrastive Learning with Information Purification and Balanced Learning

[![Zenodo Code](https://zenodo.org/badge/DOI/10.5281/zenodo.21669742.svg)](https://doi.org/10.5281/zenodo.21669742)
[![HuggingFace Data](https://img.shields.io/badge/🤗%20HuggingFace-Data-blue)](https://huggingface.co/datasets/lanyu1/dcpbst-data)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20HuggingFace-Weights-orange)](https://huggingface.co/lanyu1/dcpbst-image-pth)
[![DOI Data](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9765-green)](https://doi.org/10.57967/hf/9765)

---

## Overview

**DCPBST** is a deep learning framework for spatial domain identification in spatial transcriptomics. It integrates **gene expression** (scRNA-seq), **histopathological image features** (H&E), and **spatial coordinates** through a unified architecture that combines variational autoencoding, graph contrastive learning, and cross-modal attention mechanisms.

This repository provides the **official reproducibility package** containing all model code, configuration files, and downstream analysis scripts needed to reproduce the main experimental results from the manuscript.

---

## Data & Weights (on Hugging Face)

All datasets and pre-trained model weights are hosted on Hugging Face and can be downloaded **without authentication**:

### Pre-trained Model Weights

| Repository | Contents |
|------------|----------|
| 🤗 [lanyu1/dcpbst-image-pth](https://huggingface.co/lanyu1/dcpbst-image-pth) | `vit256_small_dino.pth`, `vit4k_xs_dino.pth` |

**Quick download:**
```bash
mkdir -p checkpoints
wget https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit256_small_dino.pth -P checkpoints/
wget https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit4k_xs_dino.pth  -P checkpoints/
```

```python
from huggingface_hub import hf_hub_download
hf_hub_download("lanyu1/dcpbst-image-pth", "vit256_small_dino.pth", local_dir="checkpoints/")
hf_hub_download("lanyu1/dcpbst-image-pth", "vit4k_xs_dino.pth",  local_dir="checkpoints/")
```

### Preprocessed Datasets

| Repository | Contents | DOI |
|------------|----------|-----|
| 🤗 [lanyu1/dcpbst-data](https://huggingface.co/datasets/lanyu1/dcpbst-data) | All 5 datasets (DLPFC ×12 slices, BRAC, PDAC, Mouse Hypothalamus, Mouse AREI) | [10.57967/hf/9765](https://doi.org/10.57967/hf/9765) |

**Quick download (entire dataset repo, ~10.4 GB):**
```bash
git clone https://huggingface.co/datasets/lanyu1/dcpbst-data data/
```

---

## Dataset Information

| Dataset | Modality | Sections / Spots | Data Source |
|---------|----------|------------------|-------------|
| **DLPFC** | 10x Visium (RNA + H&E) | 12 sections (151507–151676) | Maynard et al. 2021, *Nature Neuroscience* |
| **BRAC** | 10x Visium (RNA + H&E) | 1 section (Human Breast Cancer) | 10x Genomics |
| **PDAC** | 10x Visium (RNA + H&E) | 1 section (GSE111672) | Moncada et al. 2020, *Cell* |
| **Mouse Hypothalamus** | MERFISH (RNA + coords) | 1 section | Moffitt et al. (MERFISH 0.14) |
| **Mouse AREI / Brain Anterior** | 10x Visium (RNA + H&E) | 1 section | 10x Genomics |

### Ground Truth Labels

- **DLPFC**: `layer_guess` column in `{section_id}/metadata.tsv`
- **BRAC**: `fine_annot_type` column in `Human_breast/metadata.tsv`
- **PDAC**: loaded from label CSV in `PDAC3036911/`
- **Mouse Hypothalamus**: `ground_truth` inside the MERFISH `.h5ad` `obs`
- **Mouse AREI**: `ground_truth` column in `Mouse_Brain_Anterior/metadata.tsv`

---

## Repository Structure

```
DCPBST/
├── README.md
├── notebooks/                       # Reproducible Jupyter notebooks
│   ├── dcpbst_DLPFC_151509.ipynb       # DLPFC 151509 training + clustering
│   ├── dcpbst_DLPFC_151671.ipynb       # DLPFC 151671 training + clustering
│   ├── dcpbst_BRAC.ipynb               # BRAC training + clustering
│   ├── dcpbst_BRAC_GO_enrichment_and_violin.ipynb  # BRAC GO enrichment + violin
│   ├── dcpbst_BRAC_sensitivity.ipynb   # BRAC hyperparameter sensitivity
│   ├── dcpbst_PDAC.ipynb               # PDAC training + clustering
│   ├── dcpbst_PDAC_GO_SVG.ipynb        # PDAC DEG + GO + SVG
│   ├── dcpbst_Mouse_Hypothalamus.ipynb # MERFISH Mouse Hypothalamus
│   └── dcpbst_Mouse_AREI.ipynb         # Mouse AREI / Brain Anterior
├── ablation/                        # Ablation study scripts
│   ├── run_dlpfc_ablation.py           # DLPFC ablation (12 sections batch)
│   ├── run_brca_ablation.py            # BRAC 7-module ablation (full pipeline)
│   ├── plot_brca_ablation_bar.py       # BRAC ablation bar plot
│   ├── plot_brca_ablation_hbar.py      # BRAC ablation horizontal bar
│   └── plot_brca_ablation_hbars2.py     # BRAC ablation horizontal bar v2
├── dcpbst_package/                  # Core DCPBST package
│   ├── model.py                     # Unified configurable DCPBST model
│   ├── model_827.py                 # Config example: DLPFC / BRAC / AREI
│   ├── model_827_copy.py            # Config example: PDAC
│   ├── model_829_copy.py            # Config example: MERFISH
│   ├── preprocess.py                # Data loading & preprocessing
│   ├── utils.py                     # Helper functions
│   ├── evals.py                     # Metrics (ARI, NMI, Purity, ...)
│   ├── hist_features.py             # H&E feature extraction
│   ├── nets.py                      # GCN / GAT modules
│   ├── vision_transformer.py        # ViT for 256×256 patches
│   ├── vision_transformer4k.py      # ViT for 4K patches
│   ├── hipt_4k.py                   # HIPT 4K model
│   ├── hipt_model_utils.py          # HIPT utilities
│   └── __init__.py
├── configs/                         # Per-dataset YAML configurations
│   ├── dlpfc_config.yaml
│   ├── brac_config.yaml
│   ├── pdac_config.yaml
│   ├── hypothalamus_config.yaml
│   └── mouse_arei_config.yaml
├── downstream_analysis/             # Downstream scripts & curated data
│   ├── differential_gene_analysis.py    # DEG (Wilcoxon + Mann-Whitney)
│   ├── export_metascape_deg_list.py      # Metascape enrichment export
│   ├── plot_baseline_boxplot.py          # DLPFC baseline boxplot
│   ├── plot_hyperparameter_sensitivity.py  # Sensitivity plotting
│   ├── pdac_deg/                         # Curated PDAC DEG results
│   │   ├── deg_3_vs_4_genes_pdac.txt
│   │   └── deg_3_vs_4_with_stats_pdac.tsv
│   └── website_inputs/                   # BRAC inputs for web platform
├── figures/                         # Pre-rendered BRAC GO/KEGG figures
├── docs/
│   ├── environment_dcpbst.yml       # Conda environment
│   └── requirements_pip.txt         # pip freeze
└── legacy_models/                   # Archived model files (reference only)
```

> **Note**: `data/` and `checkpoints/` are **not** included in this repository — download them from Hugging Face as described above and place them in the repo root.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/sadnm/DCPBST.git
cd DCPBST

# 2. Download data (~10 GB) and model weights (~60 MB)
git clone https://huggingface.co/datasets/lanyu1/dcpbst-data data/
wget https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit256_small_dino.pth -P checkpoints/
wget https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit4k_xs_dino.pth  -P checkpoints/

# 3. Create conda environment
conda env create -f docs/environment_dcpbst.yml
conda activate dcpbst

# 4. Run a notebook
jupyter notebook notebooks/dcpbst_DLPFC_151509.ipynb
```

All notebooks and scripts auto-detect the repository root — **no hardcoded absolute paths**.

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.7.13 | Core language |
| PyTorch | 1.13.1 | Deep learning |
| torch-geometric | 2.3.1 | Graph neural networks |
| scanpy | 1.9.1 | Single-cell analysis |
| anndata | 0.8.0 | Annotated matrices |
| scikit-learn | 1.0.2 | Clustering & metrics |
| opencv-python | 4.6 | Image processing |
| rpy2 | 3.5.17 | R interface (mclust) |

Full pin list: [`docs/requirements_pip.txt`](docs/requirements_pip.txt).

**Hardware**: NVIDIA GPU recommended (16 GB+ RAM minimum).

---

## Method Overview

1. **RNA Preprocessing** — 3000 HVGs (seurat_v3), normalize, log1p
2. **Image Features** — ResNet or ViT (DINO pre-trained) patches
3. **Spatial Graph** — KNN graph from spatial coordinates
4. **DCPBST Training** — VAE + InfoNCE + DGI + Cross-Attention
5. **Clustering** — PCA(50) → mclust (R, `EEE`) → spatial smoothing
6. **Evaluation** — ARI / NMI / Purity / Homogeneity / Completeness / V_Measure

### Configurations

| Config | Latent | Embed | Heads | Neighbors |
|--------|--------|-------|-------|-----------|
| DLPFC / BRAC / AREI | 1024 | 512 | 8 | 7 |
| PDAC | 1024 | 256 | 16 | 7 |
| MERFISH Hypothalamus | 1024 | 256 | 8 | 6 |

### Reproducibility

All notebooks pin seeds (Python `100`, R `set.seed(2024)`). Minor variation across GPU/CUDA versions is expected — use the exact conda env (`docs/environment_dcpbst.yml`) for best reproducibility.

---

## Downstream Analysis

### Differential Gene Expression

```bash
python downstream_analysis/differential_gene_analysis.py saved_results/your_clustered.h5ad
```

Thresholds: `|log2FC| ≥ 2`, adjusted p-value < 0.05.

### GO / KEGG Enrichment

Web platform (free, no login): [https://cute-companion-liart.vercel.app](https://cute-companion-liart.vercel.app)

- BRAC pre-rendered figures + raw inputs: `figures/BRAC_downanalysis/` + `downstream_analysis/website_inputs/`
- PDAC curated DEG: `downstream_analysis/pdac_deg/`

### Ablation Study

```bash
# Full DCPBST vs 7-module ablation on BRAC
python ablation/run_brca_ablation.py --device cuda

# 12-section DLPFC batch ablation
python ablation/run_dlpfc_ablation.py --device cuda --methods dcpbst --methods imbalance_regulation
```

> **Note**: Three original ablation variants (`model_katz_quchu`, `model_qurongyu_xiaorong`, `model_qurongyu_zhuyili`) are conceptual and not included in this package — the FullModel baseline runs as-is.

---

## Citation

If you use this code, data, or weights, please cite:

```bibtex
@article{dcpbst2025,
  title={Deciphering Spatial Domains from Spatial Transcriptomics using a Dual Graph Contrastive Learning with Information Purification and Balanced Learning},
  author={...},
  journal={...},
  year={2025}
}
```

**Archived DOI**:
- Code: [10.5281/zenodo.21669742](https://doi.org/10.5281/zenodo.21669742)
- Data: [10.57967/hf/9765](https://doi.org/10.57967/hf/9765)

---

## License

Released under the **MIT License**.
