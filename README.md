# DCPBST: Deciphering Spatial Domains from Spatial Transcriptomics via Dual Graph Contrastive Learning with Information Purification and Balanced Learning

[![HuggingFace Data](https://img.shields.io/badge/🤗%20HuggingFace-Data-blue)](https://huggingface.co/datasets/lanyu1/dcpbst-data)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20HuggingFace-Weights-orange)](https://huggingface.co/lanyu1/dcpbst-image-pth)
[![DOI Data](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9765-green)](https://doi.org/10.57967/hf/9765)

---

## Overview

**DCPBST** is a deep learning framework for spatial domain identification in spatial transcriptomics. It integrates **gene expression** (scRNA-seq), **histopathological image features** (H&E), and **spatial coordinates** through a unified architecture that combines variational autoencoding, graph contrastive learning, and cross-modal attention mechanisms.

This repository provides the **official reproducibility package** containing all model code and downstream analysis scripts needed to reproduce the main experimental results from the manuscript. Each notebook is self-contained: model hyperparameters are passed directly in code (no external configuration files are read).

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
| 🤗 [lanyu1/dcpbst-data](https://huggingface.co/datasets/lanyu1/dcpbst-data) | All 5 datasets (DLPFC ×12 slices, BRCA, PDAC, Mouse Hypothalamus, Mouse AREI) | [10.57967/hf/9765](https://doi.org/10.57967/hf/9765) |

**Quick download (entire dataset repo, ~10.4 GB):**
```bash
git clone https://huggingface.co/datasets/lanyu1/dcpbst-data data/
```

---

## Dataset Information

| Dataset | Modality | Sections / Spots | Data Source |
|---------|----------|------------------|-------------|
| **DLPFC** | 10x Visium (RNA + H&E) | 12 sections (151507–151676) | Maynard et al. 2021, *Nature Neuroscience* |
| **BRCA** | 10x Visium (RNA + H&E) | 1 section (Human Breast Cancer) | 10x Genomics |
| **PDAC** | 10x Visium (RNA + H&E) | 1 section (GSE111672) | Moncada et al. 2020, *Nature Biotechnology* |
| **Mouse Hypothalamus** | MERFISH (RNA + coords) | 1 section | Moffitt et al. (MERFISH 0.14) |
| **Mouse AREI / Brain Anterior** | 10x Visium (RNA + H&E) | 1 section | 10x Genomics |

### Ground Truth Labels

- **DLPFC**: `layer_guess` column in `{section_id}/metadata.tsv`
- **BRCA**: `fine_annot_type` column in `Human_breast/metadata.tsv`
- **PDAC**: loaded from label CSV in `PDAC3036911/`
- **Mouse Hypothalamus**: `ground_truth` inside the MERFISH `.h5ad` `obs`
- **Mouse AREI**: `ground_truth` column in `Mouse_Brain_Anterior/metadata.tsv`

---

## Repository Structure

```
DCPBST/
├── README.md
├── notebooks/                       # Reproducible Jupyter notebooks (spatial-domain identification only)
│   ├── dcpbst_DLPFC_151509.ipynb       # DLPFC 151509 training + clustering
│   ├── dcpbst_DLPFC_151671.ipynb       # DLPFC 151671 training + clustering
│   ├── dcpbst_BRAC.ipynb               # BRCA training + clustering (filename kept as-is)
│   ├── dcpbst_PDAC.ipynb               # PDAC training + clustering
│   ├── dcpbst_Mouse_Hypothalamus.ipynb # MERFISH Mouse Hypothalamus
│   └── dcpbst_Mouse_AREI.ipynb         # Mouse AREI / Brain Anterior
├── ablation/                        # Ablation study scripts (BRCA only)
│   ├── run_brca_ablation.py            # BRCA 7-module ablation (full pipeline)
│   ├── plot_brca_ablation_bar.py       # BRCA ablation bar plot
│   ├── plot_brca_ablation_hbar.py      # BRCA ablation horizontal bar
│   └── plot_brca_ablation_hbars2.py    # BRCA ablation horizontal bar v2
├── dcpbst_package/                  # Core DCPBST package (imported by all notebooks/scripts)
│   ├── model_827.py                 # Model used by DLPFC & Mouse AREI notebooks
│   ├── model_827_copy.py            # Model used by BRCA & PDAC notebooks
│   ├── model_829_copy.py            # Model used by Mouse Hypothalamus notebook
│   ├── preprocess.py                # Data loading & preprocessing
│   ├── utils.py                     # Helper functions
│   ├── evals.py                     # Metrics (ARI, NMI, Purity, ...)
│   ├── hist_features.py             # H&E feature extraction
│   ├── nets.py                      # GCN / GAT modules
│   ├── ot_utils.py                  # Optimal-transport utilities
│   ├── vision_transformer.py        # ViT for 256×256 patches
│   ├── vision_transformer4k.py      # ViT for 4K patches
│   ├── hipt_4k.py                   # HIPT 4K model
│   ├── hipt_model_utils.py          # HIPT utilities
│   └── __init__.py
├── downstream_analysis/             # Downstream task scripts (DEG / GO / sensitivity)
│   ├── deg_common.py                    # Shared DEG/GO/plotting helpers (imported by the two drivers)
│   ├── run_brca_deg.py                  # BRCA: web-platform export(12vs14) + DEG(12vs14) + heatmap + volcano + GO
│   ├── run_pdac_deg.py                  # PDAC: web-platform export(3,4) + DEG(3vs4) + heatmap + volcano + GO
│   ├── run_brca_sensitivity.py          # BRCA hyperparameter sensitivity (w_kl, w_dgi, diagnose)
│   ├── plot_hyperparameter_sensitivity.py  # Sensitivity plotting
│   └── deg_results/                     # Generated tables/figures (created on first run)
│       ├── hsa_go_mapping.csv           # Human GO annotation cache (auto-downloaded, shared)
│       ├── brca/                        # BRCA outputs (incl. webtool_*.txt raw expression export)
│       └── pdac/                        # PDAC outputs (incl. webtool_*.txt raw expression export)
├── saved_results/                   # Notebook outputs (generated locally; not shipped)
├── figures/                         # Pre-rendered BRCA GO/KEGG figures
│   └── BRCA_downanalysis/
├── docs/
│   ├── EXPERIMENT_CONFIG.md            # Per-dataset hyperparameters & expected outputs
│   ├── DEVICE_EXPERIMENT_CONFIG_RTX4090.md  # RTX 4090 cross-device verification report
│   ├── environment_dcpbst.yml          # Conda environment
│   └── requirements_pip.txt            # pip freeze
└── LICENSE                          # MIT License
```

> **Note**: `data/` and `checkpoints/` are **not** included in this repository — download them from Hugging Face as described above and place them in the repo root. `saved_results/` (trained weights, clustered h5ad, metrics) and `downstream_analysis/deg_results/` (DEG/GO tables and figures) are **generated by running the notebooks / downstream scripts** and are also not shipped — only the directory names are retained.

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

### Notebook outputs (no pre-trained DCPBST weights needed)

Each notebook trains DCPBST **from scratch** — the only weights downloaded externally are the DINO ViT image encoders (`checkpoints/` above). After training, every notebook saves its own results into `saved_results/` (a local output directory; its contents are generated by running the notebooks and are not shipped with the repository):

| Output | Contents |
|--------|----------|
| `dcpbst_<DATASET>_model.pth` | Trained model parameters (`torch.save(model.state_dict(), ...)`, e.g. `dcpbst_PDAC_model.pth` written by `dcpbst_PDAC.ipynb`) |
| `dcpbst_<DATASET>_adata_with_clusters.h5ad` | AnnData with predicted spatial domains in `adata.obs['domain']` |
| `dcpbst_<DATASET>_clustering_metrics.csv` | ARI / NMI / Purity / Homogeneity / Completeness / V-measure |

The downstream DEG scripts consume only the `*_adata_with_clusters.h5ad` files — **no GPU and no model weights are required** to reproduce the differential-expression and GO-enrichment results.

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

> **Note**: clustering additionally requires **R >= 4.0** with the `mclust` R package installed (called via `rpy2`); all other steps run in pure Python.

**Computational environment**. The experiments were run on a server equipped with an Intel (R) Core (TM) i5-14600KF CPU @ 5.3GHz (14Core 6P+8E) processor, 64GB DDR4 RAM (32G*2), and NVIDIA GeForce RTX 4060Ti (16GB) GPU. The operating system used was Ubuntu 20.04 LTS. All experimental code was executed within a Python 3.7 environment.

To verify the stability of our model on different devices, all experiments were additionally re-run and the reproducible code was organized and uploaded on a second server equipped with two Intel(R) Xeon(R) Silver 4210 CPUs @ 2.20GHz (10Core each), 128GB DDR4 RAM (32G*4), 480G SSD + 16T (4T*4) storage, and an NVIDIA GeForce RTX 4090 (24GB) GPU. The operating system was Ubuntu 20.04 LTS, with the same Python 3.7 environment (PyTorch 1.13.1, CUDA 11.7). Consistent training, clustering and downstream-analysis results were obtained on both devices, confirming the stability and reproducibility of our model across hardware platforms.

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
| DLPFC / BRCA / AREI | 1024 | 512 | 8 | 7 |
| PDAC | 1024 | 256 | 16 | 7 |
| MERFISH Hypothalamus | 1024 | 256 | 8 | 6 |

### Reproducibility

All notebooks pin seeds (Python `100`, R `set.seed(2024)`). Minor variation across GPU/CUDA versions is expected — use the exact conda env (`docs/environment_dcpbst.yml`) for best reproducibility.

---

## Downstream Analysis

The notebooks only demonstrate spatial-domain identification. All other
reviewer-facing tasks (differentially expressed genes, heatmaps, volcano
plots, GO enrichment) are reproduced by two standalone driver scripts that
share the same analysis code in `downstream_analysis/deg_common.py`. They
read the clustered AnnData saved by the notebooks and need **no GPU**:

```bash
# PDAC (GSE111672): web-platform export (domains 3 and 4) + DEG 3 vs 4
#                   + heatmap + volcano + GO enrichment
python downstream_analysis/run_pdac_deg.py

# BRCA: web-platform export (domains 12 vs 14) + DEG 12 vs 14 + heatmap
#       + two volcano plots + GO enrichment
python downstream_analysis/run_brca_deg.py

# Offline / fast runs (skip GO annotation download; web export is quick):
python downstream_analysis/run_pdac_deg.py --skip-go
python downstream_analysis/run_brca_deg.py --skip-go --skip-webtool
```

Thresholds: `|log2FC| ≥ 2`, p-value < 0.05 (Wilcoxon rank-sum, scanpy).

Outputs are written to `downstream_analysis/deg_results/{brca,pdac}/`:

| Output | Description |
|--------|-------------|
| `DEG_<case>vs<ref>_all_genes.csv` / `..._significant.csv` | Full and significant DEG tables |
| `heatmap_top10_DEGs_*.pdf` | Top-10 DEG cluster-mean heatmap |
| `volcano_*.pdf` | Volcano plot(s) (BRCA: case + reference perspectives) |
| `go_enrichment_*_all_results.csv` / `..._significant.csv` | Hypergeometric GO enrichment, BH-adjusted |
| `go_bubble_*.pdf` | Top-10 GO terms per category (BP/CC/MF) |
| `webtool_data_heatmap.txt` / `webtool_sample_class.txt` | The single kept raw gene-expression export per dataset (genes × spots matrix + sample groups; BRCA: domains 12 vs 14, PDAC: domains 3 and 4) |

> **GO annotation cache**: on first GO run the human GOA annotation
> (`goa_human.gaf.gz`, ~15 MB) is downloaded once and cached as
> `downstream_analysis/deg_results/hsa_go_mapping.csv`, shared by both
> scripts. Use `--skip-go` for fully offline runs.

### Web enrichment platform

The `webtool_*.txt` files of either dataset can be uploaded to the free,
no-login web platform
[https://cute-companion-liart.vercel.app](https://cute-companion-liart.vercel.app).
Pre-rendered BRCA GO/KEGG figures are kept in `figures/BRCA_downanalysis/`.

### Ablation Study

```bash
# Full DCPBST vs 7-module ablation on BRCA
python ablation/run_brca_ablation.py --device cuda
```

> **Note**: Three original ablation variants (`model_katz_quchu`, `model_qurongyu_xiaorong`, `model_qurongyu_zhuyili`) are conceptual and not included in this package — the FullModel baseline runs as-is.

---

## Citation

If you use this code, data, or weights, please cite the manuscript (the BibTeX entry will be finalized with the author list and journal upon publication):

```bibtex
@article{dcpbst2025,
  title={Deciphering Spatial Domains from Spatial Transcriptomics using a Dual Graph Contrastive Learning with Information Purification and Balanced Learning},
  author={...},
  journal={...},
  year={2025}
}
```

**Archived DOI**:
- Data: [10.57967/hf/9765](https://doi.org/10.57967/hf/9765)

The re-analyzed public datasets originate from:
- Maynard et al. (2021), *Nature Neuroscience* 24:425–436 — DLPFC 10x Visium sections
- Moncada et al. (2020), *Nature Biotechnology* 38:333–342 ([10.1038/s41587-019-0392-8](https://doi.org/10.1038/s41587-019-0392-8)) — PDAC (GSE111672)
- Moffitt et al. (2018), *Science* 362:eaau5324 ([10.1126/science.aau5324](https://doi.org/10.1126/science.aau5324)) — MERFISH mouse hypothalamus

---

## Contributing

This repository accompanies the manuscript and is provided for reproducibility. Bug reports and questions are welcome through GitHub Issues; please include the dataset name, the command/notebook cell, and the full error message.

---

## License

Released under the **MIT License** — see the [LICENSE](LICENSE) file for details.
