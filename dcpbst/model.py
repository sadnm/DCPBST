from .nets import *
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from .utils import calculate_affinity
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.metrics.pairwise import euclidean_distances
from scanpy.external.tl import phenograph
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse import kron
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import PCA
from PIL import Image
import scipy
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import ot
import scipy.sparse as sp
from .preprocess import *
from torch.nn import MSELoss
from typing import Callable, Tuple
from torch_geometric.nn.inits import reset, uniform
import torch.nn.init as init
import torch
import torch.nn as nn
from math import sqrt
from copy import deepcopy
from tqdm import tqdm
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_p=0.2):
        """
        Encoder with two hidden layers, batch normalization and dropout.
        
        Args:
            input_dim (int): Input dimension.
            latent_dim (int): Latent space dimension.
            dropout_p (float): Dropout probability, default is 0.2.
        """
        super(Encoder, self).__init__()

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
            nn.Dropout(p=dropout_p)
        )

        self.fc_mu = nn.Linear(hidden_dim_2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim_2, latent_dim)

    def re_parametrize(self, mu, log_var):
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Forward propagation
        """
        h = self.encoder(x)
        
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        z = self.re_parametrize(mu, log_var)
        
        return mu, log_var, z

    
class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 output_dim: int, 
                 dropout_p: float = 0.2):
        """
        Decoder symmetric to encoder.
        Network structure: latent -> 512 -> 1024 -> output
        """
        super(Decoder, self).__init__()

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
            nn.Dropout(p=dropout_p)
        )

        self.final_layer = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, z):
        h = self.decoder(z)
        reconstruction = self.final_layer(h)
        return reconstruction


def z_KLD(mu, log_var):
    kld_loss=torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    return kld_loss


def pca(array):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=256, random_state=2024)
    pca_array = pca.fit_transform(array)
    return pca_array

def preprocess_adj(adj):
    e = np.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_normalized = adj.toarray() + e
    return adj_normalized


def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated


def permutation_ratio(feature, p_ratio):
    ids = np.arange(feature.shape[0])
    num_to_shuffle = int(len(ids) * p_ratio)

    shuffle_ids = np.random.permutation(ids)[:num_to_shuffle]
    feature_permutated = feature.clone()

    feature_permutated[shuffle_ids] = feature[np.random.permutation(shuffle_ids)]

    return feature_permutated


def katz_similarity(adjacency_matrix, beta):
    adjacency_matrix = np.array(adjacency_matrix, dtype='float64')

    import scipy.sparse.linalg as lg

    katz_similarity_matrix = np.linalg.inv(np.eye(adjacency_matrix.shape[0]) - beta * adjacency_matrix) - np.eye(
        adjacency_matrix.shape[0])

    return katz_similarity_matrix

def get_katz(adata):
    adj = adata.obsm['adj_spot_s']
    beta = 0.1
    katz = katz_similarity(adj, beta)
    return pca(katz)


def get_spot_feature(adata):
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        feat = adata.X.toarray()[:, ]
    else:
        feat = adata.X[:, ]

    adata.obsm['feat_spot'] = feat


def info_nce_loss(self, p, p1, p2, temp=0.05):
        loss_ctr = self.info_nce(p, p1, p2, temperature=temp)
        return loss_ctr


class InfoNCE(nn.Module):
    def __init__(self, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys, temperature):
        return info_nce(query, positive_key, negative_keys,
                        temperature=temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired'):
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
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:

        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:

        logits = query @ transpose(positive_key)

        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


EPS = 1e-15


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = msk.unsqueeze(-1) 
            sum_hidden = torch.sum(seq * msk, 1)
            num_nodes = torch.sum(msk, 1)
            return sum_hidden / (num_nodes + EPS)


class dgi(torch.nn.Module):

    """
    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """
    def __init__(
        self,
        hidden_channels: int,
        readout: Callable,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.readout = readout
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        uniform(self.hidden_channels, self.weight)

    def discriminate(self, z, summary,sigmoid= True):
        """Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (torch.Tensor): The latent space.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, pos_z, neg_z, msk=None):
        """
        Computes the DGI loss.

        Args:
            pos_z (torch.Tensor): Embeddings of positive samples (original graph nodes). 
                                  Shape: [batch_size, num_nodes, hidden_channels]
            neg_z (torch.Tensor): Embeddings of negative samples (corrupted graph nodes).
                                  Shape: [batch_size, num_nodes, hidden_channels]
            msk (torch.Tensor, optional): A mask indicating valid nodes. 
                                          Shape: [batch_size, num_nodes]. Defaults to None.

        Returns:
            torch.Tensor: The computed DGI loss (a scalar value).
        """
        summary = self.readout(pos_z, msk)
        pos_scores = self.discriminate(pos_z, summary, sigmoid=True)
        neg_scores = self.discriminate(neg_z, summary, sigmoid=True)
        if msk is not None:
           
            msk = msk.float()
            pos_loss = -torch.log(pos_scores + EPS) * msk
            neg_loss = -torch.log(1 - neg_scores + EPS) * msk
            total_loss = (pos_loss.sum() + neg_loss.sum()) / msk.sum()
        else:
            pos_loss = -torch.log(pos_scores + EPS).mean()
            neg_loss = -torch.log(1 - neg_scores + EPS).mean()
            total_loss = pos_loss + neg_loss
            
        return total_loss


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projection for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: [batch_size, tgt_len, embed_dim] - from decoder
        key:   [batch_size, src_len, embed_dim] - from encoder
        value: [batch_size, src_len, embed_dim] - from encoder
        mask:  [batch_size, tgt_len, src_len] (optional)
        """

        B, T_q, _ = query.size()
        T_k = key.size(1)

        # Project Q, K, V
        Q = self.q_proj(query)  # [B, T_q, embed_dim]
        K = self.k_proj(key)    # [B, T_k, embed_dim]
        V = self.v_proj(value)  # [B, T_k, embed_dim]

        # Split into heads
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, T_q, head_dim]
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, T_k, head_dim]
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, T_k, head_dim]

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, T_q, T_k]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # [B, heads, T_q, T_k]
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # [B, heads, T_q, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)  # [B, T_q, embed_dim]

        output = self.out_proj(context)  # [B, T_q, embed_dim]
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention module.

        Args:
        embed_dim (int): Dimension of input features.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout ratio.
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward propagation.

        Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, embed_dim].
        mask (Tensor, optional): Mask tensor for ignoring certain positions.
        """
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
        output = self.out_proj(output)

        return output

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """Single layer graph convolutional network
        
        Complete GCN function
        f = sigma(D^-1/2 A D^-1/2 * H * W)
        Convolution formula = D^-1/2 A D^-1/2 * H * W

        adjacency = D^-1/2 A D^-1/2 is already normalized, standardized Laplacian matrix
        
        Args:
            input_dim: int
                Dimension of node input features
            output_dim: int
                Output feature dimension
            use_bias : bool, optional
                Whether to use bias
        """
        super(GCN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self,input_feature,adjacency):
        """Adjacency matrix is sparse, so use sparse matrix multiplication
    
        Args: 
            adjacency: torch.sparse.FloatTensor
                Adjacency matrix
            input_feature: torch.Tensor
                Input features
        """
        adjacency = adjacency.to_sparse() if not isinstance(adjacency, torch.sparse.Tensor) else adjacency
        support = torch.matmul(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)

        if self.use_bias:
            output += self.bias
        return output
    
class GCNModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCNModel, self).__init__()
        self.gc1 = GCN(nfeat, nhid)
        self.gc2 = GCN(nhid, nclass) 
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
def scipy_sparse_to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor, and ensure type is float32."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )

    values = torch.from_numpy(sparse_mx.data).float()
    
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)


def _build_symmetric_knn_graph(feature_matrix, n_neighbors):
    """
    An efficient helper function to build symmetric k-NN adjacency matrix.
    """
    adj = kneighbors_graph(
        feature_matrix, 
        n_neighbors=n_neighbors, 
        mode='connectivity', 
        include_self=False
    )
    
    adj = adj + adj.T
    adj = (adj > 1).astype(int) + (adj == 1).astype(int)
    
    return adj.toarray()

def construct_interaction(adata, n_neighbors=6):
    """
    (Optimized version) Build k-NN graphs for spatial, gene, and image features separately.
    """
    print("Building adjacency matrices for spatial, gene, and image features...")
    
    spatial_coords = adata.obsm['spatial']
    adj_s = _build_symmetric_knn_graph(spatial_coords, n_neighbors)
    adata.obsm['adj_spot_s'] = adj_s

    if 'feat_spot' in adata.obsm:
        spot_features = adata.obsm['feat_spot']
        adj_x = _build_symmetric_knn_graph(spot_features, n_neighbors)
        adata.obsm['adj_spot_x'] = adj_x
    else:
        print("Warning: 'feat_spot' not found in adata.obsm, skipping gene expression graph construction.")

    if 'feat_img' in adata.obsm:
        image_features = adata.obsm['feat_img']
        adj_i = _build_symmetric_knn_graph(image_features, n_neighbors)
        adata.obsm['adj_spot_i'] = adj_i
    else:
        print("Warning: 'feat_img' not found in adata.obsm, skipping image feature graph construction.")
    print("All adjacency matrices construction completed.")


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_layer = nn.Linear(dim * 2, dim)

    def forward(self, x_emb, s_emb):
        gate = torch.sigmoid(self.gate_layer(torch.cat([x_emb, s_emb], dim=1)))
        return gate * x_emb + (1 - gate) * s_emb



class ModalRobustnessDiagnoser:
    def __init__(self, 
                 encoders: nn.ModuleList,
                 data_types: list,
                 device: str = 'cpu',
                 n_clusters: int = 10,
                 noise_level: float = 0.1,
                 dropout_rate: float = 0.1):
        print("Initializing Modal Robustness Diagnoser...")
        self.encoders = encoders
        self.data_types = data_types
        self.device = device
        self.n_clusters = n_clusters
        self.noise_level = noise_level
        self.dropout_rate = dropout_rate

        self.initial_encoder_sds = [deepcopy(encoder.state_dict()) for encoder in self.encoders]
        print("Initial weights of encoders have been saved.")

    def _augment_data(self, data: torch.Tensor, data_type: str) -> torch.Tensor:
    
        if data_type == 'image':
        
            original_shape = data.shape
            if len(original_shape) == 2: # [Spots, Features] -> [Spots, 1, H, W]
               
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
            
    def _diagnose_modality(self, z_orig: torch.Tensor, z_aug: torch.Tensor) -> float:
       
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
    def run(self, original_data_list: list) -> list:
    
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

class DCPBST(nn.Module):
    def __init__(self, features, sparse=False,
                 neighbors=None, device='cpu', labels=None, latent_dim=1024, # latent_dim for VAEs
                 n_clusters=None, final_feature=256, adata=None):
        super(DCPBST, self).__init__()
        self.device = device
        self.latent_dim=latent_dim
        self.num_views = len(features)
        self.features = [torch.Tensor(i) for i in features] 
        self.sparse = sparse
        self.n_clusters = n_clusters
        features = [StandardScaler().fit_transform(i) for i in features]
        self.pcs = [torch.Tensor(i).to(self.device) for i in features] 
        self.adata=adata 
      
        self.y_true = None
        if "Ground Truth" in self.adata.obs:
            print("发现 'Ground Truth'，准备用于监督学习的标签...")
            labels_encoded, unique_labels = pd.factorize(self.adata.obs["Ground Truth"])
            self.y_true = torch.tensor(labels_encoded, dtype=torch.long).to(self.device)
            self.proxies = nn.Parameter(torch.empty([self.n_clusters, latent_dim * 2]))
            nn.init.xavier_uniform_(self.proxies, gain=1.0)
        else:
            print("警告: 未在 adata.obs 中发现 'Ground Truth'。模型只能在无监督模式下运行。")


        if self.n_clusters is not None: 
             self.register_buffer('cluster_centers', None)
             self.cluster_centers_initialized = False

     
        if 'feat_spot' not in self.adata.obsm.keys():
            get_spot_feature(self.adata)

        construct_interaction(self.adata)
        self.adj_i=self.adata.obsm['adj_spot_i'] 
        self.adj_s=self.adata.obsm['adj_spot_s'] 
        self.adj_x=self.adata.obsm['adj_spot_x'] 
        self.katz = get_katz(self.adata)
        
        self.embed_dim = 256 
        self.num_heads = 16   
        self.projection_layers = nn.ModuleList([
            nn.Linear(latent_dim, self.embed_dim) for i in range(self.num_views)
        ])
    
        self.katz_projection = nn.Linear(self.katz.shape[1], self.embed_dim)

        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(self.embed_dim, self.num_heads) for _ in range(self.num_views + 1) 
        ])
       
        self.output_dim=256

        self.gcn_for_x = GCNModel(
            nfeat=self.embed_dim,  
            nhid=128,               
            nclass=self.output_dim,           
            dropout=0.5
        )
        self.gcn_for_s = GCNModel(
            nfeat=self.adata.obsm['feat_spot'].shape[1], 
            nhid=128, 
            nclass=self.output_dim, 
            dropout=0.5
        )
        self.read = AvgReadout()##全局sumaary
        self.sigm = nn.Sigmoid()
        self.info_nce = InfoNCE()
        self.dgi_module = dgi(hidden_channels=self.output_dim, readout=AvgReadout())
        self.fusion=GatedFusion(dim=self.output_dim)
        self.spatial_s = torch.from_numpy(self.adata.obsm['feat_spot']).float().to(self.device)
        self.spatial_s.requires_grad = True  
    
        ##这部分是计算最后的分类结果
        if self.n_clusters is not None:
            print(f"Classifier created: Input dim={self.output_dim}, Output dim={self.n_clusters}")
            self.classifier = nn.Linear(self.output_dim, self.n_clusters)
        else:
            self.classifier = None
       
      
        self.encoders = nn.ModuleList([Encoder(input_dim=self.pcs[i].shape[1], latent_dim=latent_dim).to(self.device) for i in range(self.num_views)])
        self.decoders = nn.ModuleList([Decoder(latent_dim=latent_dim, output_dim=self.pcs[i].shape[1]).to(self.device) for i in range(self.num_views)])
        self.diagnoser = ModalRobustnessDiagnoser(
            encoders=self.encoders,
            data_types=['gene', 'image'],
            device=self.device,
            n_clusters=self.n_clusters if self.n_clusters is not None else 10 
        )
        
        self.to(device)
        self.to(device)

    def encoder_proxies(self):
            """Extract the mean and standard deviation from the shared prototype parameters"""
            mu_proxy = self.proxies[:, :self.latent_dim]
            sigma_proxy = F.softplus(self.proxies[:, self.latent_dim:])
            return mu_proxy, sigma_proxy

    def compute_supervised_proxy_loss(self, z, y_true, temperature=0.07):
        """Calculate the supervised proxy loss between the latent feature z of a single view and the shared prototype"""
        mu_proxy, sigma_proxy = self.encoder_proxies()
        
        # Sample to obtain the prototype center
        eps_proxy = torch.randn_like(mu_proxy).unsqueeze(1).repeat(1, 10, 1) 
        z_proxy_samples = mu_proxy.unsqueeze(1) + sigma_proxy.unsqueeze(1) * eps_proxy
        z_proxy = torch.mean(z_proxy_samples, dim=1) # (num_classes, z_dim)

        att = F.cosine_similarity(z.unsqueeze(1), z_proxy.unsqueeze(0), dim=-1) # (batch_size, num_classes)
        
        # Create masks for positive and negative samples
        mask = torch.zeros_like(att, dtype=torch.bool).to(self.device)
        mask[torch.arange(att.size(0)), y_true.long()] = True

        att_positive = att.gather(1, y_true.unsqueeze(1)).squeeze(1)  # [B]

        negative_mask = torch.ones_like(att, dtype=torch.bool)
        negative_mask.scatter_(1, y_true.unsqueeze(1), False)
        att_negative = att[negative_mask].view(att.size(0), -1)  # [B, C-1]

        proxy_loss = -(att_positive - att_negative.mean(dim=1)).mean()
        logits = att / temperature  
        loss = F.cross_entropy(logits, y_true)
        return proxy_loss+loss

    def _initialize_cluster_centers(self, features_np):

        print("Initializing cluster centers with KMeans for unsupervised learning...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)  # Explicit n_init
        kmeans.fit(features_np)
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        self.cluster_centers_initialized = True
        print("Cluster centers initialized.")

    def compute_unsupervised_clustering_loss(self, features):
        if not self.cluster_centers_initialized:
            self._initialize_cluster_centers(features.detach().cpu().numpy())
        dist = torch.cdist(features, self.cluster_centers, p=2).pow(2)  # (batch_size, n_clusters)
        min_dist, _ = torch.min(dist, dim=1)
        clustering_loss = torch.mean(min_dist)
        return clustering_loss

    def update_cluster_centers(self, features, min_dist_indices):
        for i in range(self.n_clusters):    
            cluster_points = features[min_dist_indices == i]
            if len(cluster_points) > 0:
                self.cluster_centers[i] = torch.mean(cluster_points, dim=0)

    def _normalize_adj(self, adj_scipy):
        adj_scipy = adj_scipy + sp.eye(adj_scipy.shape[0]) # Add self-loops
        rowsum = np.array(adj_scipy.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj_scipy.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
    def forward(self,y_true=None):
        view_emb = []
        recon_loss = torch.tensor(0.0).to(self.device)
        kl_loss = torch.tensor(0.0).to(self.device)
        proxy_loss = torch.tensor(0.0, device=self.device)
        clustering_loss = torch.tensor(0.0, device=self.device)

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
        katz = torch.from_numpy(self.katz).float().to(self.device)
        final_feature = view_emb +[katz]
        projected_features = []
        for i in range(self.num_views):
            projected_features.append(self.projection_layers[i](final_feature[i]))
        projected_features.append(self.katz_projection(final_feature[-1]))
        stacked_features = torch.stack(projected_features, dim=1) # Shape: [N, num_total_views, D]
        global_feature = torch.mean(stacked_features, dim=1, keepdim=True) # Shape: [N, 1, D] 
        
        aligned_features = []
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                query_i = projected_features[i].unsqueeze(1)  # [N, 1, D]
                query_j = projected_features[j].unsqueeze(1)  # [N, 1, D]
                key_j = projected_features[j].unsqueeze(1)    # [N, 1, D]
                value_j = projected_features[j].unsqueeze(1)  # [N, 1, D]
                key_i = projected_features[i].unsqueeze(1)
                value_i = projected_features[i].unsqueeze(1)
                aligned_view_ij = self.cross_attention_layers[i](query_i, key_j, value_j)
                aligned_view_ji = self.cross_attention_layers[j](query_j, key_i, value_i)
                aligned_features.append(aligned_view_ij.squeeze(1))
                aligned_features.append(aligned_view_ji.squeeze(1))
        
        spatial_mask = (torch.from_numpy(self.adata.obsm['adj_spot_s']).float() > 0).to(self.device).unsqueeze(0).unsqueeze(0)
        final_embedding = torch.mean(torch.stack(aligned_features, dim=1), dim=1)

        self.adj_x=kneighbors_graph(final_embedding.detach().cpu().numpy(), n_neighbors=6, mode='connectivity')
        self.adj_x = self.adj_x + self.adj_x.T 
        self.adj_x  = scipy_sparse_to_torch_sparse(self.adj_x).to(self.device)

      
        self.adj=katz

        adj_s_scipy = kneighbors_graph(self.adj.detach().cpu().numpy(), 
                               n_neighbors=6, 
                               mode='connectivity', 
                               include_self=True) 
        adj_s_coo = adj_s_scipy.tocoo()
        indices = torch.from_numpy(
            np.vstack((adj_s_coo.row, adj_s_coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(adj_s_coo.data.astype(np.float32))
        shape = torch.Size(adj_s_coo.shape)

       
        self.adj_s = torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        self.spatial_s_a=permutation_ratio(self.spatial_s,p_ratio=1.0)

        x_emb=self.gcn_for_x(final_embedding,self.adj_x)
        s_emb=self.gcn_for_s(self.spatial_s,self.adj_s)
        s_a_emb=self.gcn_for_s(self.spatial_s_a,self.adj_s)

        info_loss=self.info_nce (query=x_emb, positive_key=s_emb, negative_keys=None,temperature=0.05)##除了样本对应的视图外 其余都是负样本

        msk = None 
        dgi_loss = self.dgi_module(s_emb.unsqueeze(0), s_a_emb.unsqueeze(0), msk)

        # final_embedding = torch.cat([x_emb, s_emb], dim=1) 

        final_embedding =self.fusion(x_emb, s_emb)


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
            "final_embedding": final_embedding
        }

    def fit(self,epochs=900, lr=1e-3, w_cls=10.0,w_recon=10.0, w_kl=0.1, w_pro=2.0,w_info=1.0, w_dgi=0.1,w_clu=1.0,diagnose_every_n_epochs=20):
        """
        Trains the DCPBST model.
        Args:
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            w_recon (float): Weight for reconstruction loss.
            w_kl (float): Weight for KL divergence loss.
            w_info (float): Weight for InfoNCE loss.
            w_dgi (float): Weight for DGI loss.
            w_pro (float): Weight for Proxy loss.
            w_clu (float): Weight for Clustering loss.(self.y_true is None)
            diagnose_every_n_epochs (int): Number of epochs between diagnoses.
        """
        super().train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion_classification = nn.CrossEntropyLoss()
        for epoch in tqdm(range(epochs), desc='Training DCPBST Model'):
           
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
                total_loss = (w_recon * recon_loss + 
                            w_kl * kl_loss + 
                            w_info * info_nce_loss + 
                            w_dgi * dgi_loss+
                            w_pro*proxy_loss+w_cls*loss_classification)
                
                total_loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f'\nEpoch {epoch+1}/{epochs} | '
                        f'Total Loss: {total_loss.item():.4f} | '
                        f'Recon: {recon_loss.item():.4f} | '
                        f'KL: {kl_loss.item():.4f} | '
                        f'InfoNCE: {info_nce_loss.item():.4f} | '
                        f'DGI: {dgi_loss.item():.4f}|'
                        f'Proxy: {proxy_loss.item():.4f}|'
                        f'Classfity: {loss_classification.item():.4f}'
                         )
                if (epoch + 1) % diagnose_every_n_epochs == 0 and epoch > 0:
                    print(f"\n--- Running Diagnose & Re-learn at epoch {epoch+1} ---")
                    alphas = self.diagnoser.run(self.pcs)
                    alpha_logs = [f"View {i}: {a:.4f}" for i, a in enumerate(alphas)]
                    print(f"Diagnose complete. Alphas: {', '.join(alpha_logs)}")
                    print("----------------------------------------------------")
            else:
                loss_dict = self(y_true=None)
                recon_loss = loss_dict["recon_loss"]
                kl_loss = loss_dict["kl_loss"]
                info_nce_loss = loss_dict["info_nce_loss"]
                dgi_loss = loss_dict["dgi_loss"]
                clustering_loss = loss_dict["clustering_loss"]

                total_loss = (w_recon * recon_loss + 
                            w_kl * kl_loss + 
                            w_info * info_nce_loss + 
                            w_dgi * dgi_loss+
                            w_clu*clustering_loss)
                
                total_loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f'\nEpoch {epoch+1}/{epochs} | '
                        f'Total Loss: {total_loss.item():.4f} | '
                        f'Recon: {recon_loss.item():.4f} | '
                        f'KL: {kl_loss.item():.4f} | '
                        f'InfoNCE: {info_nce_loss.item():.4f} | '
                        f'DGI: {dgi_loss.item():.4f}|'
                        f'Cluster: {clustering_loss.item():.4f}')
                if (epoch + 1) % diagnose_every_n_epochs == 0 and epoch > 0:
                    print(f"\n--- Running Diagnose & Re-learn at epoch {epoch+1} ---")
                    alphas = self.diagnoser.run(self.pcs)
                    alpha_logs = [f"View {i}: {a:.4f}" for i, a in enumerate(alphas)]
                    print(f"Diagnose complete. Alphas: {', '.join(alpha_logs)}")
                    print("----------------------------------------------------")
                    
        self.eval()
        print("Training finished.")
       
        self.eval()
        with torch.no_grad():
            if self.y_true is not None:
                outputs = self(y_true=self.y_true)
            else:
                outputs = self(y_true=None)
                            
            final_embedding = outputs["final_embedding"]
       
        embedding_for_clustering = final_embedding.cpu().numpy() 
        return embedding_for_clustering
    
    def cluster(self, n_clusters: int = None):
        if n_clusters is None:
            n_clusters = self.n_clusters

        self.eval()

        with torch.no_grad():
            outputs = self()
            final_embedding = outputs["final_embedding"]

        embedding_for_clustering = final_embedding.cpu().numpy() 
        clusters = KMeans(n_clusters=n_clusters, random_state=100, n_init=10).fit_predict(embedding_for_clustering)
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
