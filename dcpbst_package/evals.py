import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_mutual_info_score, v_measure_score, homogeneity_score, completeness_score
)
from pingouin import intraclass_corr
from typing import Union, Optional

def save_results_to_csv(results_dict, output_path):
    df = pd.DataFrame([results_dict])
    df.to_csv(output_path, index=False)

    
from sklearn.metrics import pairwise_distances

def compute_intra_cluster_distance(X, labels):
    unique_labels = np.unique(labels)
    intra_dists = []

    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) < 2:
            continue
        dists = pairwise_distances(cluster_points)
        mean_dist = np.sum(dists) / (len(cluster_points) * (len(cluster_points) - 1))
        intra_dists.append(mean_dist)

    return np.mean(intra_dists) if intra_dists else np.nan


def evaluate_clustering(
    X: Union[np.ndarray, pd.DataFrame],
    cluster_labels: Union[np.ndarray, list],
    reference_labels: Optional[Union[np.ndarray, list]] = None,
    method: str = 'all'
):
    """
    Evaluate clustering results using unsupervised metrics (and optional pseudo-supervised ones).

    Parameters:
    - X: Feature matrix (cells × features)
    - cluster_labels: Cluster labels predicted by your model
    - reference_labels: Optional true or pseudo labels (e.g., image clusters)
    - method: 'all', or choose from 'silhouette', 'dbi', 'ch', 'icc', 'ami', 'vmeasure'

    Returns:
    - A dictionary with metric names and values
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    cluster_labels = np.array(cluster_labels)

    results = {}

    if method in ('all', 'silhouette'):
        try:
            results['Silhouette'] = silhouette_score(X, cluster_labels)
        except:
            results['Silhouette'] = np.nan

    if method in ('all', 'dbi'):
        try:
            results['Davies-Bouldin'] = davies_bouldin_score(X, cluster_labels)
        except:
            results['Davies-Bouldin'] = np.nan

    if method in ('all', 'ch'):
        try:
            results['Calinski-Harabasz'] = calinski_harabasz_score(X, cluster_labels)
        except:
            results['Calinski-Harabasz'] = np.nan

    if method in ('all', 'icc'):
        try:
            results['ICC'] = compute_icc(X, cluster_labels)
        except:
            results['ICC'] = np.nan

    if reference_labels is not None:
        reference_labels = np.array(reference_labels)
        if method in ('all', 'ami'):
            results['Adjusted MI'] = adjusted_mutual_info_score(reference_labels, cluster_labels)
        if method in ('all', 'vmeasure'):
            results['V-Measure'] = v_measure_score(reference_labels, cluster_labels)
        if method == 'all':
            results['Homogeneity'] = homogeneity_score(reference_labels, cluster_labels)
            results['Completeness'] = completeness_score(reference_labels, cluster_labels)

    return results