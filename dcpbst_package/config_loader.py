"""
Configuration loader for DCPBST experiments.

Provides utilities to load per-dataset configuration files and
create model instances with the correct parameters.
"""
import os
import yaml
from typing import Dict, Any, Optional


CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a dataset configuration from a YAML file.

    Args:
        config_name: Name of the config file (without .yaml extension).
                     Options: 'dlpfc', 'brac', 'pdac', 'hypothalamus', 'mouse_arei'

    Returns:
        Dictionary containing all configuration parameters.

    Example:
        >>> from dcpbst_package.config_loader import load_config
        >>> config = load_config('dlpfc')
        >>> model = Dcpbst([scrna, image_emb], config=config, device='cuda')
    """
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.yaml")
    if not os.path.exists(config_path):
        available = [f.replace('.yaml', '') for f in os.listdir(CONFIGS_DIR)
                     if f.endswith('.yaml')]
        raise FileNotFoundError(
            f"Config '{config_name}' not found. Available: {available}"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_available_configs():
    """List available configuration files."""
    if not os.path.exists(CONFIGS_DIR):
        return []
    return [f.replace('.yaml', '') for f in os.listdir(CONFIGS_DIR)
            if f.endswith('.yaml')]


def create_model_from_config(features: list, config_name: str,
                              device: str = 'cpu', adata=None,
                              n_clusters: int = None):
    """
    Create a Dcpbst model instance using a preset dataset configuration.

    Args:
        features: List of feature tensors [scrna, image_emb] or [scrna, spatial]
        config_name: Name of the dataset config (e.g., 'dlpfc', 'pdac')
        device: Computing device ('cpu' or 'cuda')
        adata: AnnData object (required for graph construction)
        n_clusters: Number of clusters (auto-detected if None)

    Returns:
        Configured Dcpbst model instance.

    Example:
        >>> from dcpbst_package.config_loader import create_model_from_config
        >>> model = create_model_from_config([scrna, image_emb], 'dlpfc',
        ...                                  device='cuda', adata=adata)
        >>> embedding = model.fit()
    """
    from .model import Dcpbst

    config = load_config(config_name)

    if n_clusters is None and adata is not None:
        label_col = config.get('label_col_name', 'Ground Truth')
        if label_col in adata.obs:
            n_clusters = len(set(adata.obs[label_col]))
        else:
            n_clusters = config.get('n_clusters', 7)

    model_kwargs = {
        'sparse': config.get('sparse', False),
        'neighbors': config.get('neighbors', 7),
        'device': device,
        'latent_dim': config.get('latent_dim', 1024),
        'n_clusters': n_clusters,
        'adata': adata,
    }

    model = Dcpbst(features, **model_kwargs)

    # Store config for reference
    model._config = config
    model._config_name = config_name

    return model