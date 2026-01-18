import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import scipy
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import kneighbors_graph
from PIL import Image
import random
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
    homogeneity_completeness_v_measure, calinski_harabasz_score, davies_bouldin_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
input_dim = 1433 
hidden1_dim = 32
hidden2_dim = 16
use_feature = True


def construct_interaction(adata, n_neighbors=6, x=0.05):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # sim = normalize_vector(cal_weight_matrix(adata, x=x))
    # distance_matrix = sim * distance_matrix

    interaction_s = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction_s[i, y] = 1
    graph_neg = np.ones([n_spot, n_spot]) - interaction_s

    adata.obsm['graph_neigh'] = interaction_s
    adata.obsm['graph_neg'] = graph_neg

    adj_spot = interaction_s
    adj_spot = adj_spot + adj_spot.T
    adj_spot = np.where(adj_spot > 1, 1, adj_spot)
    adata.obsm['adj_spot_s'] = adj_spot

    spot_feat = adata.obsm['feat_spot']
    # sim = cal_weight_matrix(adata, x)

    spot_feat = adata.obsm['feat_spot']
    spot_dis = ot.dist(spot_feat, spot_feat, metric='euclidean')
    interaction_x = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = spot_dis[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction_x[i, y] = 1
    graph_neg = np.ones([n_spot, n_spot]) - interaction_x

    adata.obsm['graph_neigh'] = interaction_x
    adata.obsm['graph_neg'] = graph_neg

    adj_spot = interaction_x
    adj_spot = adj_spot + adj_spot.T
    adj_spot = np.where(adj_spot > 1, 1, adj_spot)
    adata.obsm['adj_spot_x'] = adj_spot

    img_feat = adata.obsm['feat_img']
    img_dis = ot.dist(img_feat, img_feat, metric='euclidean')
    interaction_i = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = img_dis[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction_i[i, y] = 1
    graph_neg = np.ones([n_spot, n_spot]) - interaction_i

    adata.obsm['graph_neigh'] = interaction_i
    adata.obsm['graph_neg'] = graph_neg

    adj_spot = interaction_i
    adj_spot = adj_spot + adj_spot.T
    adj_spot = np.where(adj_spot > 1, 1, adj_spot)
    adata.obsm['adj_spot_i'] = adj_spot


class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), hidden2_dim)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self,adj):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred
		

def protein_norm(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)


def preprocess(adata,modality):
  adata.var_names_make_unique()
  if modality in ['rna','atac']:
    sc.pp.filter_genes(adata,min_cells=10)
    sc.pp.log1p(adata)

    if scipy.sparse.issparse(adata.X):
      return adata.X.A
    else:
      return adata.X

  elif modality=='protein':
    adata.X = np.apply_along_axis(protein_norm, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X)))
    return adata.X     

  elif modality=='metabolite':
    sc.pp.log1p(adata)
    if scipy.sparse.issparse(adata.X):
      return adata.X.A
    else:
      return adata.X


def calculate_affinity(X1, sig=30, sparse = False, neighbors = 100):
  if not sparse:
    dist1 = pairwise_distances(X1)
    a1 = np.exp(-1*(dist1**2)/(2*(sig**2)))
    return a1
  else:
    dist1 = kneighbors_graph(X1, n_neighbors = neighbors, mode='distance')
    dist1.data = np.exp(-1*(dist1.data**2)/(2*(sig**2)))
    dist1.eliminate_zeros()
    return dist1

def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = x % 20
    x = (x // 10) + (x % 10) * 2
    return cmap(x)



def cmap_tab30(x):
    n_base = 20
    n_max = 30
    brightness = 0.7
    brightness = (brightness,) * 3 + (1.0,)
    isin_base = (x < n_base)[..., np.newaxis]
    isin_extended = ((x >= n_base) * (x < n_max))[..., np.newaxis]
    isin_beyond = (x >= n_max)[..., np.newaxis]
    color = (
        isin_base * cmap_tab20(x)
        + isin_extended * cmap_tab20(x-n_base) * brightness
        + isin_beyond * (0.0, 0.0, 0.0, 1.0))
    return color


def cmap_tab70(x):
    cmap_base = cmap_tab30
    brightness = 0.5
    brightness = np.array([brightness] * 3 + [1.0])
    color = [
        cmap_base(x),  # same as base colormap
        1 - (1 - cmap_base(x-20)) * brightness,  # brighter
        cmap_base(x-20) * brightness,  # darker
        1 - (1 - cmap_base(x-40)) * brightness**2,  # even brighter
        cmap_base(x-40) * brightness**2,  # even darker
        [0.0, 0.0, 0.0, 1.0],  # black
        ]
    x = x[..., np.newaxis]
    isin = [
        (x < 30),
        (x >= 30) * (x < 40),
        (x >= 40) * (x < 50),
        (x >= 50) * (x < 60),
        (x >= 60) * (x < 70),
        (x >= 70)]
    color_out = np.sum(
            [isi * col for isi, col in zip(isin, color)],
            axis=0)
    return color_out


def plot(clusters,locs):
  locs['2'] = locs['2'].astype('int')
  locs['3'] = locs['3'].astype('int')
  im1 = np.empty((locs['2'].max()+1, locs['3'].max()+1))
  im1[:] = np.nan
  im1[locs['2'],locs['3']] = clusters
  im2 = cmap_tab70(im1.astype('int'))
  im2[np.isnan(im1)] = 1
  im3 = Image.fromarray((im2 * 255).astype(np.uint8))
  return im3

def plot_on_histology(clusters, locs, im, scale, s=10, title=None):
  locs = locs*scale
  locs = locs.round().astype('int')
  im = im[(locs['4'].min()-10):(locs['4'].max()+10),(locs['5'].min()-10):(locs['5'].max()+10)]
  locs = locs-locs.min()+10
  cmap1 = mcolors.ListedColormap([cmap_tab70(np.array(i)) for i in range(len(np.unique(clusters)))])
  plt.imshow(im, alpha=0.7)
  plot = plt.scatter(x=locs['5'], y=locs['4'], c = clusters, cmap=cmap1, s=s)
  if title:
        plt.title(title)
  plt.axis('off')
  return plot

def set_random_seed(seed=100):
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)




def normalize_vector(arr):
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)

    normalized_arr = (arr - min_vals) / (max_vals - min_vals)

    return normalized_arr


def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def calculate_clustering_matrix(df, y_pred, ground_truth, sample, methods_):
    ari = adjusted_rand_score(y_pred, ground_truth)
    nmi = normalized_mutual_info_score(y_pred, ground_truth)
    purity = purity_score(y_pred, ground_truth)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_pred, ground_truth)
    print("ARI={}, NMI={},Purity={},V_Measure={}".format(ari, nmi,purity,v_measure))

    df = df._append(pd.Series([sample, ari, nmi, purity, homogeneity, completeness, v_measure, methods_],
                             index=['Sample', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure',
                                    'methods']), ignore_index=True)
    return df

def calculate_parameter_matrix(df, alpha, beta, y_pred, ground_truth, sample, methods_):
    ari = adjusted_rand_score(y_pred, ground_truth)
    nmi = normalized_mutual_info_score(y_pred, ground_truth)
    purity = purity_score(y_pred, ground_truth)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_pred, ground_truth)
    print(ari)

    df = df._append(pd.Series([sample, alpha, beta, ari, nmi, purity, homogeneity, completeness, v_measure, methods_],
                             index=['Sample', 'Alpha', 'Beta', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure',
                                    'methods']), ignore_index=True)
    return df

def calculate_one_parameter_matrix(df, parameter, y_pred, ground_truth, sample, methods_):
    ari = adjusted_rand_score(y_pred, ground_truth)
    nmi = normalized_mutual_info_score(y_pred, ground_truth)
    purity = purity_score(y_pred, ground_truth)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_pred, ground_truth)
    print(ari)

    df = df._append(pd.Series([sample, parameter, ari, nmi, purity, homogeneity, completeness, v_measure, methods_],
                             index=['Sample', 'Parameter', 'ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness', 'V_Measure',
                                    'methods']), ignore_index=True)
    return df


def nolabel_clustering_matrix(df, X, labels, sample, methods_):
    sc_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    df = df._append(pd.Series([sc_score, ch_score, db_score, sample, methods_],
                             index=['Silhouette-Coefficient', 'Calinski-Harabasz', 'Davies-Bouldin', 'Sample',
                                    'methods']),
                   ignore_index=True)

    return df


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2024):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    print(res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01,
               refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1.
    end : float
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """

    pca = PCA(n_components=20, random_state=2024)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding

    # adata.obsm["X_umap"] = get_umap(embedding)

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters).fit(embedding)
        kmeans_result = [i + 1 for i in kmeans.labels_]
        adata.obs['domain'] = list(map(lambda x: str(x), kmeans_result))
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data1.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res
