## DCPBST: Deciphering spatial domains from spatial transcriptomics using a dual graph contrastive learning with information purification and balanced learning

### Pre-trained Models

Download the pre-trained model weights from Hugging Face:

| Model | Description | Download Link |
|-------|-------------|---------------|
| ViT-256 Small DINO | Vision Transformer for 256x256 patches | [vit256_small_dino.pth](https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit256_small_dino.pth) |
| ViT-4k XS DINO | Vision Transformer for 4k resolution | [vit4k_xs_dino.pth](https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit4k_xs_dino.pth) |

**Quick Download using wget:**
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download models
wget https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit256_small_dino.pth -P checkpoints/
wget https://huggingface.co/lanyu1/dcpbst-image-pth/resolve/main/vit4k_xs_dino.pth -P checkpoints/
```

**Quick Download using Python:**
```python
from huggingface_hub import hf_hub_download
import os

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# Download models
hf_hub_download(repo_id="lanyu1/dcpbst-image-pth", 
                filename="vit256_small_dino.pth",
                local_dir="checkpoints/",
                local_dir_use_symlinks=False)
hf_hub_download(repo_id="lanyu1/dcpbst-image-pth", 
                filename="vit4k_xs_dino.pth",
                local_dir="checkpoints/",
                local_dir_use_symlinks=False)
```

###  Download data

#### Download from Hugging Face (Recommended)

The preprocessed datasets used in this study are publicly hosted on Hugging Face:

**Repository:** [lanyu1/dcpbst-data](https://huggingface.co/datasets/lanyu1/dcpbst-data) (Public)
**DOI:** [10.57967/hf/9765](https://doi.org/10.57967/hf/9765)

The repository contains the preprocessed data for all five datasets analysed in the paper:
the twelve human dorsolateral prefrontal cortex (DLPFC) slices (`151507`–`151676`), the human
breast cancer section (`Human_breast`), the mouse anterior brain section (`Mouse_Brain_Anterior`),
and the human pancreatic ductal adenocarcinoma section (`PDAC3036911`). Each Visium directory
provides the gene-expression matrix (`filtered_feature_bc_matrix.h5`), the manual annotations
(`metadata.tsv`) and the `spatial/` folder with the histology image and coordinates.

No authentication is required.

**Quick Download using Python:**
```python
from huggingface_hub import hf_hub_download
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Download specific dataset (example: 151509)
# Download h5 file
hf_hub_download(repo_id="lanyu1/dcpbst-data", 
                filename="151509/filtered_feature_bc_matrix.h5",
                repo_type="dataset",
                local_dir="data/",
                local_dir_use_symlinks=False)

# Download metadata
hf_hub_download(repo_id="lanyu1/dcpbst-data", 
                filename="151509/metadata.tsv",
                repo_type="dataset",
                local_dir="data/",
                local_dir_use_symlinks=False)

# Download spatial files
hf_hub_download(repo_id="lanyu1/dcpbst-data", 
                filename="151509/spatial/scalefactors_json.json",
                repo_type="dataset",
                local_dir="data/",
                local_dir_use_symlinks=False)
hf_hub_download(repo_id="lanyu1/dcpbst-data", 
                filename="151509/spatial/tissue_positions_list.csv",
                repo_type="dataset",
                local_dir="data/",
                local_dir_use_symlinks=False)
```

**Using Git LFS (for batch download):**
```bash
# Install git-lfs if not already installed
# Run: git lfs install

# Clone the entire dataset repository (~10.4 GB)
git clone https://huggingface.co/datasets/lanyu1/dcpbst-data data/
```

#### Original Data Sources

The unprocessed raw versions of all datasets can be obtained directly from their original sources:

| Dataset                                        | Link                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Human dorsolateral prefrontal cortex dataset   | https://github.com/JiangBioLab/DeepST                          |
| Mouse brain serial section (sagittal-anterior) | https://support.10xgenomics.com/spatial-gene-expression/datasets |
| Human breast cancer                            | https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1 |
| Human Pancreatic Ductal Adenocarcinoma         | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111672 |
| Mouse hypothalamic preoptic dataset            | http://sdmbench.drai.cn                                      |

### Citation

The source code of this study is archived on Zenodo:

**Code DOI:** [10.5281/zenodo.21669780](https://doi.org/10.5281/zenodo.21669780)
**Data DOI:** [10.57967/hf/9765](https://doi.org/10.57967/hf/9765)


### Requirements
- anndata==0.8.0
- cloudpickle==2.2.1
- dask==2022.2.0
- dask-image==2022.9.0
- einops==0.6.0
- fsspec==2023.1.0
- giotto-ph==0.2.2
- giotto-tda==0.6.0
- h5py==3.8.0
- igraph==0.10.8
- imageio==2.31.2
- leidenalg==0.9.0
- llvmlite==0.39.1
- louvain==0.8.0
- matplotlib==3.5.3
- matplotlib-scalebar==0.8.1
- natsort==8.4.0
- numba==0.56.4
- numcodecs==0.10.2
- numpy==1.21.6
- omnipath==1.0.8
- opencv-python==4.6.0.66
- pandas==1.3.5
- pandas_flavor==0.7.0
- Pillow==9.5.0
- PIMS==0.7
- pingouin==0.5.3
- plotly==5.18.0
- POT==0.9.5
- pydantic==1.10.24
- pyflagser==0.4.5
- pynndescent==0.5.13
- python-igraph==0.10.8
- python-louvain==0.16
- pytz==2025.2
- PyWavelets==1.3.0
- rpy2==3.5.17
- scanpy==1.9.1
- scikit-image==0.19.3
- scikit-learn==1.0.2
- scikit-misc==0.1.4
- scipy==1.7.3
- seaborn==0.12.2
- squidpy==1.2.2
- statsmodels==0.13.5
- tifffile==2021.11.2
- torch==1.13.1
- torch-geometric==2.3.1
- torchvision==0.14.1
- umap-learn==0.5.7
- xarray==0.20.2
- zarr==2.12.0