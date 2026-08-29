"""
DCPBST: Multi-modal Integration for Spatial Omics

A unified configurable model for multi-modal spatial data integration.
Supports Visium (RNA + image features) and MERFISH (RNA + spatial coordinates) modalities.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn.inits import uniform
import torch.nn.init as init

from scipy.sparse import csr_matrix, csc_matrix
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from itertools import combinations
from copy import deepcopy
from typing import Callable

from torch.nn import Parameter

# --- Optional imports ---
try:
    import ot
    _HAS_OT = True
except ImportError:
    _HAS_OT = False

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

EPS = 1e-15


# ==============================================================================
#  Utility Functions
# ==============================================================================

def z_KLD(mu, log_var):
    """KL divergence between learned latent and standard normal prior N(0, I)."""
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    return kld_loss


def pca(array, n_components=1000):
    """Reduce dimensionality via PCA. Clamps n_components to data dimensions."""
    max_components = min(array.shape[0], array.shape[1]) - 1
    n_components = min(n_components, max_components)
    if n_components <= 0:
        n_components = 1
    pca_model = PCA(n_components=n_components, random_state=2024)
    return pca_model.fit_transform(array)


def preprocess_adj(adj):
    """Symmetrically normalize an adjacency matrix and add self-loops."""
    e = np.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray() + e


def add_contrastive_label(adata):
    """Assign contrastive labels (1 for original, 0 for corrupted) to AnnData."""
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def permutation(feature):
    """Randomly permute rows of a feature tensor (graph corruption)."""
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    return feature[ids]


def permutation_ratio(feature, p_ratio):
    """Randomly permute a fraction of rows in a feature tensor."""
    ids = np.arange(feature.shape[0])
    num_to_shuffle = int(len(ids) * p_ratio)
    shuffle_ids = np.random.permutation(ids)[:num_to_shuffle]
    feature_permutated = feature.clone()
    feature_permutated[shuffle_ids] = feature[np.random.permutation(shuffle_ids)]
    return feature_permutated


def katz_similarity(adjacency_matrix, beta):
    """Compute Katz similarity: (I - beta * A)^{-1} - I with numerical stability."""
    adjacency_matrix = np.array(adjacency_matrix, dtype='float64')
    n = adjacency_matrix.shape[0]
    A = np.eye(n) - beta * adjacency_matrix
    # Add small regularization for numerical stability
    A_reg = A + np.eye(n) * 1e-10
    katz_matrix = np.linalg.inv(A_reg) - np.eye(n)
    return katz_matrix


def get_katz(adata, beta=0.1):
    """Compute Katz similarity features from the spatial adjacency graph, then PCA."""
    adj = adata.obsm['adj_spot_s']
    katz = katz_similarity(adj, beta)
    return pca(katz)


def get_spot_feature(adata):
    """Extract spot-level features from AnnData and store in obsm."""
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        feat = adata.X.toarray()
    else:
        feat = adata.X
    adata.obsm['feat_spot'] = feat


def info_nce_loss_fn(query, positive_key, negative_keys=None, temperature=0.05):
    """Convenience wrapper around info_nce."""
    return info_nce(query, positive_key, negative_keys, temperature=temperature)


def _transpose(x):
    return x.transpose(-2, -1)


def _normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def info_nce(query, positive_key, negative_keys=None, temperature=1.0,
              reduction='mean', negative_mode='unpaired'):
    """
    InfoNCE loss for contrastive learning.

    Args:
        query: Anchor embeddings [N, D].
        positive_key: Positive embeddings [N, D].
        negative_keys: Negative embeddings, either [N, D] (unpaired) or [N, K, D] (paired).
        temperature: Softmax temperature.
        reduction: Loss reduction mode ('mean' or 'sum').
        negative_mode: How negatives are interpreted.

    Returns:
        Cross-entropy loss scalar.
    """
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', negative_keys must match query count.")
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> must have the same dimensionality.')
    if negative_keys is not None and query.shape[-1] != negative_keys.shape[-1]:
        raise ValueError('Vectors of <query> and <negative_keys> must have the same dimensionality.')

    query, positive_key, negative_keys = _normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        if negative_mode == 'unpaired':
            negative_logits = query @ _transpose(negative_keys)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ _transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        logits = query @ _transpose(positive_key)
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def scipy_sparse_to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse float32 tensor."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def _build_symmetric_knn_graph(feature_matrix, n_neighbors):
    """Build a symmetric k-NN adjacency matrix (dense)."""
    adj = kneighbors_graph(
        feature_matrix,
        n_neighbors=n_neighbors,
        mode='connectivity',
        include_self=False,
    )
    adj = adj + adj.T
    adj = (adj > 1).astype(int) + (adj == 1).astype(int)
    return adj.toarray()


def construct_interaction(adata, n_neighbors=7, use_img_features=True):
    """
    Build spot-to-spot k-NN adjacency matrices for spatial, gene, and (optionally)
    image features, storing them in adata.obsm.
    """
    print("Constructing adjacency matrices for spatial, gene, and image features...")

    spatial_coords = adata.obsm['spatial']
    adj_s = _build_symmetric_knn_graph(spatial_coords, n_neighbors)
    adata.obsm['adj_spot_s'] = adj_s

    if 'feat_spot' in adata.obsm:
        spot_features = adata.obsm['feat_spot']
        adj_x = _build_symmetric_knn_graph(spot_features, n_neighbors)
        adata.obsm['adj_spot_x'] = adj_x
    else:
        print("Warning: 'feat_spot' not found in adata.obsm, skipping gene graph.")

    if use_img_features and 'feat_img' in adata.obsm:
        image_features = adata.obsm['feat_img']
        adj_i = _build_symmetric_knn_graph(image_features, n_neighbors)
        adata.obsm['adj_spot_i'] = adj_i
    elif use_img_features:
        print("Warning: 'feat_img' not found in adata.obsm, skipping image graph.")

    print("All adjacency matrices built.")


# ==============================================================================
#  Helper Classes
# ==============================================================================

class Encoder(nn.Module):
    """VAE encoder with two hidden layers, BatchNorm, and Dropout."""

    def __init__(self, input_dim, latent_dim, dropout_p=0.2):
        super().__init__()
        hidden_dim_1 = 1024
        hidden_dim_2 = 512

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.fc_mu = nn.Linear(hidden_dim_2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim_2, latent_dim)

    def re_parametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.re_parametrize(mu, log_var)
        return mu, log_var, z


class Decoder(nn.Module):
    """VAE decoder symmetric to the encoder."""

    def __init__(self, latent_dim, output_dim, dropout_p=0.2):
        super().__init__()
        hidden_dim_1 = 512
        hidden_dim_2 = 1024

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.final_layer = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, z):
        h = self.decoder(z)
        return self.final_layer(h)


class InfoNCE(nn.Module):
    """Module wrapper around the info_nce contrastive loss."""

    def __init__(self, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None, temperature=1.0):
        return info_nce(query, positive_key, negative_keys,
                        temperature=temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


class CrossAttention(nn.Module):
    """Cross-attention for cross-modal alignment."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.size()
        T_k = key.size(1)

        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(context)


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        output = attn @ v
        output = output.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(output)


class GCN(nn.Module):
    """Single graph convolution layer: D^{-1/2} A D^{-1/2} H W."""

    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):
        if not isinstance(adjacency, torch.sparse.Tensor):
            adjacency = adjacency.to_sparse()
        support = torch.matmul(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GCNModel(nn.Module):
    """Two-layer GCN with ReLU activation and dropout."""

    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gc1 = GCN(nfeat, nhid)
        self.gc2 = GCN(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class AvgReadout(nn.Module):
    """Global average readout for DGI."""

    def __init__(self):
        super().__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = msk.unsqueeze(-1)
            sum_hidden = torch.sum(seq * msk, 1)
            num_nodes = torch.sum(msk, 1)
            return sum_hidden / (num_nodes + EPS)


class DGI(nn.Module):
    """Deep Graph Infomax mutual information maximization."""

    def __init__(self, hidden_channels, readout):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.readout = readout
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.hidden_channels, self.weight)

    def discriminate(self, z, summary, sigmoid=True):
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, pos_z, neg_z, msk=None):
        summary = self.readout(pos_z, msk)
        pos_scores = self.discriminate(pos_z, summary, sigmoid=True)
        neg_scores = self.discriminate(neg_z, summary, sigmoid=True)

        if msk is not None:
            msk = msk.float()
            pos_loss = -torch.log(pos_scores + EPS) * msk
            neg_loss = -torch.log(1 - neg_scores + EPS) * msk
            return (pos_loss.sum() + neg_loss.sum()) / msk.sum()
        else:
            pos_loss = -torch.log(pos_scores + EPS).mean()
            neg_loss = -torch.log(1 - neg_scores + EPS).mean()
            return pos_loss + neg_loss


class GatedFusion(nn.Module):
    """Gate-weighted fusion of two embedding streams."""

    def __init__(self, dim):
        super().__init__()
        self.gate_layer = nn.Linear(dim * 2, dim)

    def forward(self, x_emb, s_emb):
        gate = torch.sigmoid(self.gate_layer(torch.cat([x_emb, s_emb], dim=1)))
        return gate * x_emb + (1 - gate) * s_emb


class ModalRobustnessDiagnoser:
    """
    Diagnose modal robustness via augmentation-based cluster-stability gap,
    then re-learn encoder weights to balance modalities.
    """

    def __init__(self, encoders, data_types, device='cpu',
                 n_clusters=10, noise_level=0.1, dropout_rate=0.1):
        self.encoders = encoders
        self.data_types = data_types
        self.device = device
        self.n_clusters = n_clusters
        self.noise_level = noise_level
        self.dropout_rate = dropout_rate
        self.initial_encoder_sds = [
            deepcopy(encoder.state_dict()) for encoder in self.encoders
        ]

    def _augment_data(self, data, data_type):
        if data_type == 'image':
            original_shape = data.shape
            if len(original_shape) == 2:
                h = w = int(np.sqrt(original_shape[1]))
                if h * w == original_shape[1]:
                    data = data.reshape(original_shape[0], 1, h, w)
                else:
                    noise = torch.randn_like(data) * self.noise_level
                    return torch.clamp(data + noise, 0.0, data.max().item())
            noise = torch.randn_like(data) * self.noise_level
            noisy_data = torch.clamp(data + noise, 0.0, 1.0)
            return noisy_data.reshape(original_shape)
        elif data_type == 'gene':
            mask = torch.bernoulli(torch.full_like(data, 1 - self.dropout_rate))
            return data * mask
        else:
            mask = torch.bernoulli(torch.full_like(data, 1 - self.dropout_rate))
            return data * mask

    def _diagnose_modality(self, z_orig, z_aug):
        z_orig_np = z_orig.detach().cpu().numpy()
        z_aug_np = z_aug.detach().cpu().numpy()
        if len(z_orig_np) < self.n_clusters:
            return 0.0
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        c_orig = kmeans.fit_predict(z_orig_np)
        if len(np.unique(c_orig)) < 2:
            return 0.0
        score_orig = silhouette_score(z_orig_np, c_orig)
        score_aug = silhouette_score(z_aug_np, c_orig)
        gap = score_orig - score_aug
        return max(0, gap)

    @torch.no_grad()
    def run(self, original_data_list):
        self.encoders.eval()
        gaps = []
        for i, (encoder, data_orig) in enumerate(zip(self.encoders, original_data_list)):
            data_type = self.data_types[i]
            data_orig_device = data_orig.to(self.device)
            data_aug = self._augment_data(data_orig_device, data_type)
            mu_orig, _, _ = encoder(data_orig_device)
            mu_aug, _, _ = encoder(data_aug)
            gap = self._diagnose_modality(mu_orig, mu_aug)
            gaps.append(gap)

        total_gap = sum(gaps)
        alphas = [g / (total_gap + 1e-8) for g in gaps]

        for i, (encoder, alpha) in enumerate(zip(self.encoders, alphas)):
            initial_sd = self.initial_encoder_sds[i]
            for name, param in encoder.named_parameters():
                initial_param = initial_sd[name]
                relearned_param = (1 - alpha) * param.data + alpha * initial_param
                param.data.copy_(relearned_param)

        self.encoders.train()
        return alphas


# ==============================================================================
#  DCPBST Model
# ==============================================================================

# Default loss weights per dataset type
_DEFAULT_LOSS_WEIGHTS = {
    'dlpfc': {
        'w_cls': 10.0, 'w_recon': 10.0, 'w_kl': 0.1,
        'w_pro': 2.0, 'w_info': 3.0, 'w_dgi': 0.1, 'w_clu': 1.0,
    },
    'brac': {
        'w_cls': 10.0, 'w_recon': 10.0, 'w_kl': 0.1,
        'w_pro': 2.0, 'w_info': 3.0, 'w_dgi': 0.1, 'w_clu': 1.0,
    },
    'arei': {
        'w_cls': 10.0, 'w_recon': 10.0, 'w_kl': 0.1,
        'w_pro': 2.0, 'w_info': 3.0, 'w_dgi': 0.1, 'w_clu': 1.0,
    },
    'pdac': {
        'w_cls': 10.0, 'w_recon': 10.0, 'w_kl': 0.5,
        'w_pro': 3.0, 'w_info': 3.0, 'w_dgi': 0.1, 'w_clu': 1.0,
    },
    'merfish': {
        'w_cls': 10.0, 'w_recon': 10.0, 'w_kl': 0.1,
        'w_pro': 2.0, 'w_info': 1.0, 'w_dgi': 0.1, 'w_clu': 2.0,
    },
}


class Dcpbst(nn.Module):
    """
    Multi-modal Integration for Spatial Omics (DCPBST).

    A configurable deep generative model for integrating multiple spatial data
    modalities (e.g., RNA + image features for Visium, or RNA + spatial
    coordinates for MERFISH).

    Args:
        features (list): List of numpy arrays, one per data view.
        adata (anndata.AnnData): Annotated data matrix with .obsm['spatial'].
        config (dict or namespace, optional): Configuration dictionary. Supported keys:
            - pca_n_components (int): PCA dim, default 1000 (256 for PDAC).
            - embed_dim (int): Cross-attention embed dim, default 512 (256 for PDAC).
            - num_heads (int): Attention heads, default 8 (16 for PDAC).
            - output_dim (int): Final embedding dim, default 256.
            - latent_dim (int): VAE latent dim, default 1024.
            - neighbors (int): KNN neighbors, default 7 (6 for MERFISH).
            - use_ot (bool): Import POT library, default False.
            - spatial_s_requires_grad (bool): spatial_s requires grad, default False.
            - label_col_name (str): Label column in adata.obs, default "Ground Truth".
            - use_img_features (bool): Use image features, default True (False for MERFISH).
            - gat_dropout (float): GCN dropout, default 0.5.
            - diagnose_every_n_epochs (int): Diagnose frequency, default 20.
            - loss_weights (dict): Override per-dataset loss weights.
        device (str): Compute device, default 'cpu'.
        n_clusters (int): Number of clusters, required for supervised mode.
        sparse (bool): Whether features are sparse.
    """

    def __init__(self, features, adata, config=None, device='cpu',
                 n_clusters=None, sparse=False):
        super().__init__()

        # --- Parse config ---
        if config is None:
            config = {}
        elif hasattr(config, '__dict__'):
            config = vars(config)

        self.device = device
        self.n_clusters = n_clusters
        self.sparse = sparse
        self.adata = adata
        self.num_views = len(features)

        # Hyperparameters with defaults
        self.pca_n_components = config.get('pca_n_components', 1000)
        self.embed_dim = config.get('embed_dim', 512)
        self.num_heads = config.get('num_heads', 8)
        self.output_dim = config.get('output_dim', 256)
        self.latent_dim = config.get('latent_dim', 1024)
        self.neighbors = config.get('neighbors', 7)
        self.use_ot = config.get('use_ot', False)
        self.spatial_s_requires_grad = config.get('spatial_s_requires_grad', False)
        self.label_col_name = config.get('label_col_name', 'Ground Truth')
        self.use_img_features = config.get('use_img_features', True)
        self.gat_dropout = config.get('gat_dropout', 0.5)
        self.diagnose_every_n_epochs = config.get('diagnose_every_n_epochs', 20)

        default_weights = _DEFAULT_LOSS_WEIGHTS['dlpfc']
        custom_weights = config.get('loss_weights', {})
        self.loss_weights = {**default_weights, **custom_weights}

        # Optional OT import
        if self.use_ot and not _HAS_OT:
            print("Warning: use_ot=True but POT library not installed. Falling back to CPU.")

        # --- Preprocess features ---
        features_scaled = [StandardScaler().fit_transform(f) for f in features]
        self.pcs = [torch.Tensor(f).to(self.device) for f in features_scaled]

        # --- Labels ---
        self.y_true = None
        if self.label_col_name in self.adata.obs:
            print(f"Found '{self.label_col_name}'. Encoding labels for supervised learning...")
            labels_encoded, _ = pd.factorize(self.adata.obs[self.label_col_name])
            self.y_true = torch.tensor(labels_encoded, dtype=torch.long).to(self.device)
            self.proxies = nn.Parameter(
                torch.empty([self.n_clusters, self.latent_dim * 2])
            )
            nn.init.xavier_uniform_(self.proxies, gain=1.0)
        else:
            print(f"Warning: '{self.label_col_name}' not found in adata.obs. "
                  "Model will run in unsupervised mode.")

        # Unsupervised cluster centers
        if self.n_clusters is not None:
            self.register_buffer('cluster_centers', None)
            self.cluster_centers_initialized = False

        # --- Build interaction graphs ---
        if 'feat_spot' not in self.adata.obsm:
            get_spot_feature(self.adata)

        construct_interaction(
            self.adata,
            n_neighbors=self.neighbors,
            use_img_features=self.use_img_features,
        )

        self.adj_s = self.adata.obsm['adj_spot_s']
        self.adj_x = self.adata.obsm.get('adj_spot_x', None)
        self.katz = get_katz(self.adata)

        # --- Cross-modal alignment layers ---
        self.projection_layers = nn.ModuleList([
            nn.Linear(self.latent_dim, self.embed_dim)
            for _ in range(self.num_views)
        ])
        self.katz_projection = nn.Linear(self.katz.shape[1], self.embed_dim)

        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(self.embed_dim, self.num_heads)
            for _ in range(self.num_views + 1)
        ])

        # --- GCN branches ---
        self.gcn_for_x = GCNModel(
            nfeat=self.embed_dim, nhid=128,
            nclass=self.output_dim, dropout=self.gat_dropout,
        )

        if self.use_img_features:
            spatial_dim = self.adata.obsm['feat_spot'].shape[1]
        else:
            if isinstance(self.adata.X, csr_matrix) or isinstance(self.adata.X, csc_matrix):
                spatial_dim = self.adata.X.shape[1]
            else:
                spatial_dim = self.adata.X.shape[1]

        self.gcn_for_s = GCNModel(
            nfeat=spatial_dim, nhid=128,
            nclass=self.output_dim, dropout=self.gat_dropout,
        )

        # --- DGI, contrastive, fusion ---
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.info_nce = InfoNCE()
        self.dgi_module = DGI(
            hidden_channels=self.output_dim,
            readout=AvgReadout(),
        )
        self.fusion = GatedFusion(dim=self.output_dim)

        # Spatial features
        if self.use_img_features:
            self.spatial_s = torch.from_numpy(
                self.adata.obsm['feat_spot']
            ).float().to(self.device)
        else:
            if isinstance(self.adata.X, csr_matrix) or isinstance(self.adata.X, csc_matrix):
                self.spatial_s = torch.from_numpy(
                    self.adata.X.toarray()
                ).float().to(self.device)
            else:
                self.spatial_s = torch.from_numpy(
                    self.adata.X
                ).float().to(self.device)

        if self.spatial_s_requires_grad:
            self.spatial_s.requires_grad = True

        # --- Classifier ---
        if self.n_clusters is not None:
            print(f"Classifier created: Input dim={self.output_dim}, "
                  f"Output dim={self.n_clusters}")
            self.classifier = nn.Linear(self.output_dim, self.n_clusters)
        else:
            self.classifier = None

        # --- Encoders / Decoders ---
        self.encoders = nn.ModuleList([
            Encoder(
                input_dim=self.pcs[i].shape[1],
                latent_dim=self.latent_dim,
            ).to(self.device)
            for i in range(self.num_views)
        ])
        self.decoders = nn.ModuleList([
            Decoder(
                latent_dim=self.latent_dim,
                output_dim=self.pcs[i].shape[1],
            ).to(self.device)
            for i in range(self.num_views)
        ])

        # --- Modal robustness diagnoser ---
        data_types = ['gene'] + ['image'] * (self.num_views - 1)
        self.diagnoser = ModalRobustnessDiagnoser(
            encoders=self.encoders,
            data_types=data_types,
            device=self.device,
            n_clusters=self.n_clusters if self.n_clusters is not None else 10,
        )

        self.alpha_history = []
        self.to(device)

    # ------------------------------------------------------------------
    #  Proxy-based supervised loss
    # ------------------------------------------------------------------

    def encoder_proxies(self):
        """Extract mean and std from the shared proxy parameter."""
        mu_proxy = self.proxies[:, :self.latent_dim]
        sigma_proxy = F.softplus(self.proxies[:, self.latent_dim:])
        return mu_proxy, sigma_proxy

    def compute_supervised_proxy_loss(self, z, y_true, temperature=0.07):
        """Supervised proxy loss between latent z and shared class prototypes."""
        mu_proxy, sigma_proxy = self.encoder_proxies()

        eps_proxy = torch.randn_like(mu_proxy).unsqueeze(1).repeat(1, 10, 1)
        z_proxy_samples = mu_proxy.unsqueeze(1) + sigma_proxy.unsqueeze(1) * eps_proxy
        z_proxy = torch.mean(z_proxy_samples, dim=1)

        att = F.cosine_similarity(z.unsqueeze(1), z_proxy.unsqueeze(0), dim=-1)

        mask = torch.zeros_like(att, dtype=torch.bool).to(self.device)
        mask[torch.arange(att.size(0)), y_true.long()] = True

        att_positive = att.gather(1, y_true.unsqueeze(1)).squeeze(1)
        negative_mask = torch.ones_like(att, dtype=torch.bool)
        negative_mask.scatter_(1, y_true.unsqueeze(1), False)
        att_negative = att[negative_mask].view(att.size(0), -1)

        proxy_loss = -(att_positive - att_negative.mean(dim=1)).mean()
        logits = att / temperature
        loss = F.cross_entropy(logits, y_true)
        return proxy_loss + loss

    # ------------------------------------------------------------------
    #  Unsupervised clustering
    # ------------------------------------------------------------------

    def _initialize_cluster_centers(self, features_np):
        """Initialize cluster centers with KMeans."""
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(features_np)
        self.cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=self.device
        )
        self.cluster_centers_initialized = True

    def compute_unsupervised_clustering_loss(self, features):
        """Assign each point to its nearest cluster center and minimize distance."""
        if not self.cluster_centers_initialized:
            self._initialize_cluster_centers(features.detach().cpu().numpy())

        dist = torch.cdist(features, self.cluster_centers, p=2).pow(2)
        min_dist, _ = torch.min(dist, dim=1)
        return torch.mean(min_dist)

    # ------------------------------------------------------------------
    #  Adjacency normalization
    # ------------------------------------------------------------------

    def _normalize_adj(self, adj_scipy):
        adj_scipy = adj_scipy + sp.eye(adj_scipy.shape[0])
        rowsum = np.array(adj_scipy.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj_scipy.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, y_true=None):
        view_emb = []
        recon_loss = torch.tensor(0.0, device=self.device)
        kl_loss = torch.tensor(0.0, device=self.device)
        proxy_loss = torch.tensor(0.0, device=self.device)
        clustering_loss = torch.tensor(0.0, device=self.device)

        # --- VAE encoding / decoding ---
        for i in range(self.num_views):
            mu, logvar, z = self.encoders[i](self.pcs[i])
            x_hat = self.decoders[i](z)
            view_emb.append(z)
            recon_loss += F.mse_loss(x_hat, self.pcs[i])
            kl_loss += z_KLD(mu, logvar)
            if y_true is not None and hasattr(self, 'proxies'):
                proxy_loss += self.compute_supervised_proxy_loss(z, y_true)

        if y_true is not None and self.num_views > 0:
            proxy_loss /= self.num_views

        # --- Katz high-order spatial features ---
        katz = torch.from_numpy(self.katz).float().to(self.device)
        final_feature = view_emb + [katz]

        # --- Cross-modal alignment via cross-attention ---
        projected_features = []
        for i in range(self.num_views):
            projected_features.append(self.projection_layers[i](final_feature[i]))
        projected_features.append(self.katz_projection(final_feature[-1]))

        stacked_features = torch.stack(projected_features, dim=1)
        global_feature = torch.mean(stacked_features, dim=1, keepdim=True)

        aligned_features = []
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                query_i = projected_features[i].unsqueeze(1)
                query_j = projected_features[j].unsqueeze(1)

                aligned_view_ij = self.cross_attention_layers[i](
                    query_i, query_j, query_j
                )
                aligned_view_ji = self.cross_attention_layers[j](
                    query_j, query_i, query_i
                )
                aligned_features.append(aligned_view_ij.squeeze(1))
                aligned_features.append(aligned_view_ji.squeeze(1))

        final_embedding = torch.mean(torch.stack(aligned_features, dim=1), dim=1)

        # --- Dynamic graph construction ---
        self.adj_x = kneighbors_graph(
            final_embedding.detach().cpu().numpy(),
            n_neighbors=self.neighbors,
            mode='connectivity',
        )
        self.adj_x = self.adj_x + self.adj_x.T
        self.adj_x = scipy_sparse_to_torch_sparse(self.adj_x).to(self.device)

        # New spatial relation graph from Katz features
        self.adj = katz
        adj_s_scipy = kneighbors_graph(
            self.adj.detach().cpu().numpy(),
            n_neighbors=self.neighbors,
            mode='connectivity',
            include_self=True,
        )
        adj_s_coo = adj_s_scipy.tocoo()
        indices = torch.from_numpy(
            np.vstack((adj_s_coo.row, adj_s_coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(adj_s_coo.data.astype(np.float32))
        shape = torch.Size(adj_s_coo.shape)
        self.adj_s = torch.sparse.FloatTensor(indices, values, shape).to(self.device)

        # --- GCN branches ---
        spatial_s_a = permutation_ratio(self.spatial_s, p_ratio=1.0)
        x_emb = self.gcn_for_x(final_embedding, self.adj_x)
        s_emb = self.gcn_for_s(self.spatial_s, self.adj_s)
        s_a_emb = self.gcn_for_s(spatial_s_a, self.adj_s)

        # --- Contrastive and DGI losses ---
        info_loss = self.info_nce(
            query=x_emb, positive_key=s_emb,
            negative_keys=None, temperature=0.05,
        )
        msk = None
        dgi_loss = self.dgi_module(
            s_emb.unsqueeze(0), s_a_emb.unsqueeze(0), msk
        )

        # --- Gated fusion ---
        final_embedding = self.fusion(x_emb, s_emb)

        # --- Unsupervised clustering loss ---
        if y_true is None and self.n_clusters is not None:
            clustering_loss = self.compute_unsupervised_clustering_loss(final_embedding)

        logits = None
        if self.classifier is not None:
            logits = self.classifier(final_embedding)

        return {
            "logits": logits,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "info_nce_loss": info_loss,
            "proxy_loss": proxy_loss,
            "clustering_loss": clustering_loss,
            "dgi_loss": dgi_loss,
            "final_embedding": final_embedding,
        }

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def fit(self, epochs=900, lr=1e-3, w_cls=None, w_recon=None,
            w_kl=None, w_pro=None, w_info=None, w_dgi=None, w_clu=None,
            diagnose_every_n_epochs=None):
        """
        Train the DCPBST model.

        Args:
            epochs: Number of training epochs.
            lr: Learning rate.
            w_cls: Weight for classification loss.
            w_recon: Weight for reconstruction loss.
            w_kl: Weight for KL divergence loss.
            w_pro: Weight for proxy loss.
            w_info: Weight for InfoNCE loss.
            w_dgi: Weight for DGI loss.
            w_clu: Weight for clustering loss.
            diagnose_every_n_epochs: Run modal robustness diagnostics every N epochs.

        Returns:
            numpy.ndarray: Final embedding for downstream clustering.
        """
        # Apply config defaults for loss weights
        lw = self.loss_weights
        w_cls = w_cls if w_cls is not None else lw['w_cls']
        w_recon = w_recon if w_recon is not None else lw['w_recon']
        w_kl = w_kl if w_kl is not None else lw['w_kl']
        w_pro = w_pro if w_pro is not None else lw['w_pro']
        w_info = w_info if w_info is not None else lw['w_info']
        w_dgi = w_dgi if w_dgi is not None else lw['w_dgi']
        w_clu = w_clu if w_clu is not None else lw['w_clu']

        if diagnose_every_n_epochs is None:
            diagnose_every_n_epochs = self.diagnose_every_n_epochs

        super().train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion_classification = nn.CrossEntropyLoss()

        for epoch in tqdm(range(epochs), desc='Training Dcpbst Model'):
            optimizer.zero_grad()

            if self.y_true is not None:
                loss_dict = self(y_true=self.y_true)
                logits = loss_dict["logits"]
                recon_loss = loss_dict["recon_loss"]
                kl_loss = loss_dict["kl_loss"]
                info_nce_loss = loss_dict["info_nce_loss"]
                dgi_loss = loss_dict["dgi_loss"]
                proxy_loss = loss_dict["proxy_loss"]
                loss_classification = criterion_classification(logits, self.y_true)

                total_loss = (
                    w_recon * recon_loss +
                    w_kl * kl_loss +
                    w_info * info_nce_loss +
                    w_dgi * dgi_loss +
                    w_pro * proxy_loss +
                    w_cls * loss_classification
                )
                total_loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(
                        f'\nEpoch {epoch + 1}/{epochs} | '
                        f'Total: {total_loss.item():.4f} | '
                        f'Recon: {recon_loss.item():.4f} | '
                        f'KL: {kl_loss.item():.4f} | '
                        f'InfoNCE: {info_nce_loss.item():.4f} | '
                        f'DGI: {dgi_loss.item():.4f} | '
                        f'Proxy: {proxy_loss.item():.4f} | '
                        f'Classify: {loss_classification.item():.4f}'
                    )
            else:
                loss_dict = self(y_true=None)
                recon_loss = loss_dict["recon_loss"]
                kl_loss = loss_dict["kl_loss"]
                info_nce_loss = loss_dict["info_nce_loss"]
                dgi_loss = loss_dict["dgi_loss"]
                clustering_loss = loss_dict["clustering_loss"]

                total_loss = (
                    w_recon * recon_loss +
                    w_kl * kl_loss +
                    w_info * info_nce_loss +
                    w_dgi * dgi_loss +
                    w_clu * clustering_loss
                )
                total_loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(
                        f'\nEpoch {epoch + 1}/{epochs} | '
                        f'Total: {total_loss.item():.4f} | '
                        f'Recon: {recon_loss.item():.4f} | '
                        f'KL: {kl_loss.item():.4f} | '
                        f'InfoNCE: {info_nce_loss.item():.4f} | '
                        f'DGI: {dgi_loss.item():.4f} | '
                        f'Cluster: {clustering_loss.item():.4f}'
                    )

            # --- Modal robustness diagnostics ---
            if (epoch + 1) % diagnose_every_n_epochs == 0 and epoch > 0:
                print(f"\n--- Running Diagnose & Re-learn at epoch {epoch + 1} ---")
                alphas = self.diagnoser.run(self.pcs)
                self.alpha_history.append({
                    'epoch': epoch + 1,
                    'alphas': [float(a) for a in alphas],
                })
                alpha_logs = [f"View {i}: {a:.4f}" for i, a in enumerate(alphas)]
                print(f"Diagnose complete. Alphas: {', '.join(alpha_logs)}")
                print("----------------------------------------------------")

        # --- Inference ---
        self.eval()
        with torch.no_grad():
            outputs = self(y_true=self.y_true)
            final_embedding = outputs["final_embedding"]

        return final_embedding.cpu().numpy()

    # ------------------------------------------------------------------
    #  Clustering
    # ------------------------------------------------------------------

    def cluster(self, n_clusters=None):
        """
        Run KMeans clustering on the learned embedding and report metrics.

        Args:
            n_clusters: Number of clusters. Falls back to self.n_clusters if not given.

        Returns:
            numpy.ndarray: Cluster label assignments.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        self.eval()
        with torch.no_grad():
            outputs = self()
            final_embedding = outputs["final_embedding"]

        embedding_for_clustering = final_embedding.cpu().numpy()
        clusters = KMeans(
            n_clusters=n_clusters, random_state=100, n_init=10
        ).fit_predict(embedding_for_clustering)
        self.clusters = clusters

        if len(set(clusters)) > 1 and embedding_for_clustering.shape[0] > 1:
            sil_score = silhouette_score(embedding_for_clustering, clusters)
            ch_score = calinski_harabasz_score(embedding_for_clustering, clusters)
            db_score = davies_bouldin_score(embedding_for_clustering, clusters)
            print(f'Clustering metrics for {n_clusters} clusters:')
            print(f'  Silhouette Score: {sil_score:.4f}')
            print(f'  Calinski-Harabasz Index: {ch_score:.4f}')
            print(f'  Davies-Bouldin Index: {db_score:.4f}')
        else:
            print("Warning: Cannot compute clustering metrics (too few clusters or samples).")

        return clusters

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def get_alpha_history_for_heatmap(self):
        """Return the alpha-value history in a format suitable for heatmap plotting."""
        if not self.alpha_history:
            print("Warning: No alpha history recorded. Ensure diagnostics ran during training.")
            return {'epochs': [], 'alpha_matrix': np.array([]), 'view_names': []}

        epochs = [record['epoch'] for record in self.alpha_history]
        num_views = len(self.alpha_history[0]['alphas'])
        alpha_matrix = np.array([record['alphas'] for record in self.alpha_history])
        view_names = [f'View {i}' for i in range(num_views)]

        return {
            'epochs': epochs,
            'alpha_matrix': alpha_matrix,
            'view_names': view_names,
        }