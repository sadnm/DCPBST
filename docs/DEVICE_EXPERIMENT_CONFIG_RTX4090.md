# 实验设备与运行配置报告（RTX 4090 复现环境）

- **报告日期**: 2026-08-31
- **对应 notebook**: `notebooks/dcpbst_BRAC.ipynb`（空间域识别）；DEG/GO 下游任务由 `downstream_analysis/run_brca_deg.py` 复现
- **实验内容**: BRCA（Human Breast Cancer，10x Visium）空间域识别 + DEG/GO 富集下游分析
- **原始环境参照**: 原始作者 tutorial notebook（`tutorial_dlpfc_pdac_go_svg.ipynb`，非本仓库文件）运行时输出记录为 `CUDA is available. GPU: NVIDIA GeForce RTX 4060 Ti`；本报告记录本复现设备的对应信息，格式与该记录一致。

---

## 1. 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | **2 × NVIDIA GeForce RTX 4090**（Compute Capability 8.9，Ada Lovelace） |
| 单卡显存 | 24 GB（24564 MiB / 卡，共约 48 GB） |
| NVIDIA 驱动 | 550.163.01 |
| 驱动支持 CUDA 版本 | CUDA 12.4（驱动侧上限） |
| CPU | 2 × Intel Xeon Silver 4210 @ 2.20GHz（共 20 核 / 40 线程，睿频 3.2 GHz） |
| 内存 | 125 GB（约 98 GB 可用） |
| 系统盘 / 数据盘 | `/data` 挂载 11 TB（已用 7.0 TB，可用 3.4 TB） |
| 操作系统 | Ubuntu 20.04.6 LTS（内核 5.15.0-139-generic） |
| 运行方式 | Docker 容器（hostname: `lixiaowen`） |
| 本机 notebook 指定 GPU | **GPU 1**（通过 `CUDA_VISIBLE_DEVICES=1`，容器内可见为 `cuda:0`） |

> GPU 选择原因：BRCA 任务含 ResNet 图像 patch 特征提取（3798 张 448×448 patch）与 900 epoch 多模态训练，显存峰值较高；运行时 GPU 1 空闲显存约 17–19 GB，优于 GPU 0。

## 2. 软件环境

使用 conda 环境 **`miso`**（`/root/miniconda3/envs/miso`），已与项目参考环境清单
`/data/miso/supplement/env/conda_env_miso.txt` 完成 **305/305 包名+版本精确对齐**。

| 组件 | 版本 |
|------|------|
| Python | 3.7.12 |
| PyTorch | 1.13.1+cu117（CUDA runtime 11.7，cuDNN 8.5.0） |
| CUDA 可用性 | `torch.cuda.is_available() = True`，可见设备数 2 |
| R | 4.5.1（conda R，`lib/R`） |
| mclust（R 聚类包） | 6.1.1 |
| rpy2 | 3.5.17 |
| scanpy | 1.9.1 |
| anndata | 0.8.0 |
| numpy / pandas / scipy | 1.21.6 / 1.3.5 / 1.7.3 |
| scikit-learn | 1.0.2 |
| POT（`import ot`，最优传输） | 0.9.5 |
| h5py / Pillow / tqdm | 3.8.0 / 9.5.0 / 4.64.1 |
| Jupyter kernel | `Python (miso)`（ipykernel 6.16.2，notebook 6.5.7） |

环境修复备注（保证与参考环境一致地运行）：

- `sitecustomize.py` 在解释器启动时预加载 conda 环境的 `libstdc++/libgcc_s`（`RTLD_GLOBAL`），解决 `torch` 先于 `rpy2` 导入时系统旧版 libstdc++ 导致 `libR.so` 加载失败的问题；
- kernelspec 注入 `LD_LIBRARY_PATH=/root/miniconda3/envs/miso/lib`，并设置 `R_HOME`。

## 3. Notebook 运行配置（dcpbst_BRAC.ipynb）

### 3.1 设备与缓存开关（cell 0 / cell 1）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `CUDA_VISIBLE_DEVICES` | `'1'` | 在 `import torch` 之前设置，训练使用物理 GPU 1 |
| `FORCE_RETRAIN` | `True` | **每次运行都重新训练，不读取 `saved_results/` 下的缓存结果** |
| 随机种子 | `seed = 100` | 同时固定 `numpy` / `torch` / `random` / `cudnn`（deterministic，benchmark 关闭）；R mclust 内 `set.seed(2024)` |

### 3.2 数据

| 项目 | 值 |
|------|-----|
| 数据目录 | `data/Human_breast/`（`filtered_feature_bc_matrix.h5` + `metadata.tsv` + `spatial/`） |
| 标签列 | `fine_annot_type`（metadata.tsv，细粒度注释共 20 类） |
| 空间点数（spots） | **3798** |
| HVG 选择 | 3000（`seurat_v3`），标准化 `target_sum=1e4` + `log1p` |
| 图像 patch 缓存 | `data/cut_img_BRCA/cut_img_BRCA.npy`，形状 **(3798, 448, 448, 3)** |
| ResNet 图像特征缓存 | `data/cut_img_BRCA/Img_feat_brca.npy`，形状 **(3798, 1000)**（存在则直接加载，跳过 ResNet 提取） |
| H&E 图像加载 | 三级回退：`full_image.tif` → `tissue_hires_image.png`（坐标缩放）→ `tissue_lowres_image.png` |

### 3.3 模型与训练

| 配置项 | 值 |
|--------|-----|
| 模型 | `dcpbst_package/model_827_copy.py` 的 `Dcpbst`（两模态：RNA + 图像特征） |
| 实例化 | `Dcpbst([scrna, image_emb], sparse=False, device='cuda', n_clusters=7, adata=adata, neighbors=7)` |
| 图构建近邻数 | 7（GAT，dropout 0.5） |
| 编码器结构 | input → 1024 → 512 → latent；聚类解码器 latent → 512 → 1024 → output |
| RNA PCA 降维 | 1000 维 |
| 训练轮数 | **900 epochs** |
| 优化器 / 学习率 | Adam / `lr=1e-3` |
| 损失权重 | `w_cls=10.0`, `w_recon=10.0`, `w_kl=0.1`, `w_pro=3.0`, `w_info=1.0`, `w_dgi=0.1`, `w_clu=1.0` |
| 诊断/重学习 | 每 20 epoch 执行一次 Diagnose & Re-learn（输出双视图 alpha 权重） |
| 训练日志 | 每 10 epoch 打印 Total Loss |

### 3.4 聚类与后处理

| 配置项 | 值 |
|--------|-----|
| 聚类域数 `n_clusters` | **7** |
| 聚类方法 | R `mclust`，EEE 模型，经 `rpy2` 调用（fallback：kmeans / leiden） |
| 聚类前 PCA | 50 维（作用于训练嵌入 `obsm['emb']`） |
| 空间标签平滑 | `refine_label`，radius = 50（POT `ot.dist` 欧氏距离） |
| 评价指标 | ARI、NMI、Purity、Homogeneity、Completeness、V_Measure（`dcpbst_package/evals.py`） |
| 下游分析 | 由 `downstream_analysis/run_brca_deg.py` 复现：DEG（Wilcoxon，\|log2FC\|≥2 且 P<0.05；cluster 12 vs 14，121 个显著 DEG）→ 热图、火山图、GO 富集气泡图；另导出网页平台输入（domains 12 vs 14，626 spots：431 + 195） |

### 3.5 结果输出

训练完成后仍会写入（但 notebook 默认**不会读回**，因 cell 1 中 `FORCE_RETRAIN=True`）：

```
saved_results/dcpbst_BRCA_adata_with_clusters.h5ad   # 含 obsm['emb'] 训练嵌入 + obs['domain'] 聚类
saved_results/dcpbst_BRCA_model.pth                  # 模型 state_dict
saved_results/dcpbst_BRCA_clustering_metrics.csv     # 聚类评价指标
```

## 4. 与原始 RTX 4060 Ti 环境的对应关系

| 项目 | 原始环境（4060 Ti） | 本复现环境（4090） |
|------|--------------------|--------------------|
| GPU | NVIDIA GeForce RTX 4060 Ti（原始 notebook 输出记录） | 2 × NVIDIA GeForce RTX 4090（本 notebook 指定 GPU 1） |
| 代码包 | `miso` 包（工作目录 `/data/shijie/miso`） | `dcpbst_package`（仓库 `reproducible_experiments/`，路径自动检测） |
| 模型文件 | `miso.model_827_copy`（原始 notebook 导入） | `dcpbst_package.model_827_copy`（BRCA notebook 实际导入，与 EXPERIMENT_CONFIG.md 一致） |
| Python/PyTorch | 以参考环境清单为准 | Python 3.7.12 + PyTorch 1.13.1+cu117，305 包与 `supplement/env/conda_env_miso.txt` 精确对齐 |
| 训练超参 | epochs=900, lr=1e-3, seed=100 | 完全一致 |
| 缓存机制 | 每次重新运行 | 本 copy 设 `FORCE_RETRAIN=True`，每次重新训练（与原始行为一致） |

> 说明：RTX 4090 显存（24 GB）大于 RTX 4060 Ti，且本环境 PyTorch 使用 CUDA 11.7 runtime（驱动 550.163.01 向下兼容），训练与聚类结果不受设备型号影响；随机性由全局种子 100 / R 种子 2024 固定。

## 5. 运行须知

1. **重启内核**后选择 `Python (miso)` kernel，从头顺序运行；`CUDA_VISIBLE_DEVICES` 必须在 `import torch` 前生效。
2. 运行中查看 GPU 占用：`nvidia-smi`；本 notebook 进程应出现在 GPU 1 上。
3. 若遇 CUDA OOM：优先确认 GPU 1 无其他大显存进程；其次可减小图像 batch size / 注意力头数（本配置 `neighbors=7`，图像特征已缓存为 npy，正常峰值远低于 24 GB）。
4. 若希望恢复"训练一次、后续秒加载"的缓存复用：将 cell 1 中 `FORCE_RETRAIN = True` 改回 `False`（前提是 `saved_results/dcpbst_BRCA_adata_with_clusters.h5ad` 已生成）。
5. 训练约 900 epoch（含每 20 epoch 的 Diagnose & Re-learn），随后自动执行 mclust 聚类、空间域平滑与指标计算，并写出 `saved_results/` 下的 h5ad / model.pth / metrics；DEG、热图、火山图与 GO 富集等下游任务不在 notebook 内，统一由 `python downstream_analysis/run_brca_deg.py` 复现（仅需 CPU，消费保存的 h5ad）。
