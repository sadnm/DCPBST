# Preprocessing for gene expression data
# Graph construction utilities
# ----------------- Data preprocessing code -----------------
# Following the processing approach for multi-omics spatial data
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
####
from sklearn.preprocessing import StandardScaler


import scanpy as sc
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


# Data preprocessing
def preprocess(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    return adata[:, adata.var['highly_variable']]

# Spatial position map - spatial adjacency graph construction
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=False):
    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)

    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + G.T
    G.data = np.minimum(G.data, 1)
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.obsm['adj'] = G
    return G

# Feature similarity - feature graph constructed after PCA dimensionality reduction
def Cal_Feature_Net(adata, k=20, mode= "distance", metric="minkowski", include_self=False):
    print("Begin calculate feature graph")
    feature_graph = kneighbors_graph(adata.obsm['feat_pca'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    feature_graph = feature_graph+feature_graph.T
    feature_graph.data = np.minimum(feature_graph.data, 1)
    adata.obsm['feature_adj'] = feature_graph
    print(feature_graph)

# Construct similarity matrix using OT distance
def Cal_Feature_Net_Corrected(features_np, n_neighbors=6, metric='euclidean'):
    """
    Build a symmetric K-nearest-neighbor graph from a feature matrix.

    Args:
        features_np (np.ndarray): Feature matrix with shape (n_samples, n_features).
        n_neighbors (int): Number of neighbors to retain in the adjacency graph.
        metric (str): Distance metric to use.

    Returns:
        scipy.sparse.csr_matrix: Symmetric adjacency matrix.
    """
    # Step 1: Use scikit-learn's NearestNeighbors for efficient neighbor lookup
    # It computes the distance matrix internally and finds K-neighbors more efficiently
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='auto')
    knn.fit(features_np)
    
    # kneighbors_graph returns a sparse matrix indicating connections to k nearest neighbors
    adj = knn.kneighbors_graph(mode='connectivity')

    # Step 2: Convert to symmetric adjacency matrix
    adj_sym = adj.maximum(adj.T)

    return adj_sym.tocoo()  # Return COO format for easy conversion to torch sparse tensor


def construct_sinkhorn_graph(adata, feature_key='feat', n_neighbors=6, reg=0.1):
    X = adata.obsm[feature_key]
    n = X.shape[0]
    # Default uniform distribution
    a = b = np.ones(n) / n
    M = ot.dist(X, X, metric='euclidean')

    sinkhorn_matrix = ot.sinkhorn(a, b, M, reg=reg)
    distance_matrix = M  # Still using distance to build KNN graph

    adj = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(distance_matrix[i])[:n_neighbors+1]
        for j in neighbors:
            if i != j:
                adj[i, j] = 1

    adj_sym = adj + adj.T
    adj_sym = np.where(adj_sym > 1, 1, adj_sym)

    adata.obsm['adj_feat_sinkhorn'] = adj_sym
    adata.obsm['sinkhorn_matrix'] = sinkhorn_matrix

    return adj_sym

# Preprocess adjacency matrix
def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])  # Add self-loops
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

#     else:   
       
#     else:
    
    

def Get_feature(adata, emb, deconvolution=False):
    """
    Replace default features in `adata` with cross-view consistency embeddings.

    Parameters:
    - adata: AnnData object
    - concat_z: torch.Tensor of shape [N, num_views * latent_dim], concatenated multi-view embeddings
    - deconvolution: if True, skip HVG selection (optional)
    """
    # Step 1: Standardize the multi-view embedding    
    # Step 2: Save standardized embedding as main feature
    adata.obsm['feat'] = emb

    feat_a = np.random.permutation(emb)  # row-wise permutation

    adata.obsm['feat_a'] = feat_a

    return adata

def concat_adj(adj_1,adj_2):
    n1, m1 = adj_1.shape
    n2, m2 = adj_2.shape
    adj_1_right = sp.hstack([adj_1, sp.csr_matrix((n1, m2))])  # Fill zero matrix on right of adj_1
    adj_2_left = sp.hstack([sp.csr_matrix((n2, m1)), adj_2])  # Fill zero matrix on left of adj_2

    adj_combined = sp.vstack([adj_1_right, adj_2_left])  # Vertically stack the two matrices
    return adj_combined


def pca(adata, use_reps=None, n_comps=50):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)

    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    adata.obsm['feat_pca'] = feat_pca

#1. InfoNCE - align features and spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

import numpy as np

class Encoder_cons(Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(Encoder_cons, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()
        self.info_nce = InfoNCE()
        self.fuse = nn.Sequential(
            nn.Linear(out_features * 2, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
)
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)

    def info_nce_loss(self, p, p1, p2, temp=0.2):
        loss_1 = self.info_nce(p, p1, p2, temperature=temp)  # spatial -> feat
        loss_2 = self.info_nce(p1, p, p2, temperature=temp) # feat -> spatial (note: p and p1 swapped)
        return 0.5*(loss_1 + loss_2)
    
    def forward(self, feat, feat_a, adj_spatial, adj_feat, adj_feat_a):
        # Cross-view representation
        h = feat @ self.weight1
        h_a = feat_a @ self.weight1

        z_spatial = adj_spatial @ h
        z_feat    = adj_feat @ h
        z_feat_a  = adj_feat_a @ h_a

        emb_spatial = self.act(z_spatial)
        emb_feat    = self.act(z_feat)
        emb_feat_a  = self.act(z_feat_a)

        # InfoNCE cross-graph contrastive learning
        loss = self.info_nce_loss(emb_spatial, emb_feat, emb_feat_a)
        z_combined = self.fuse(torch.cat([emb_spatial, emb_feat], dim=1))
        return emb_spatial, emb_feat, emb_feat_a, z_combined,loss


class InfoNCE(nn.Module):
    def __init__(self, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys, temperature=0.05):
        return info_nce(query, positive_key, negative_keys, temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired'):
    if query.dim() != 2 or positive_key.dim() != 2:
        raise ValueError("query and positive_key must be 2D")

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

    if negative_keys is not None:
        negative_logits = query @ transpose(negative_keys)
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

























#1. Build cell similarity matrix based on OT distance



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a 

