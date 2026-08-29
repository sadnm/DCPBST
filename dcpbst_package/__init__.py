"""
DCPBST: Multi-modal Integration for Spatial Omics.

Core package providing the unified DCPBST model, legacy model variants,
and configuration utilities for reproducible spatial omics experiments.
"""

try:
    from .model import Dcpbst
except ImportError:
    pass

# Import legacy model variants for backward compatibility
# They accept slightly different hyperparameters per dataset
try:
    from .model_827 import Dcpbst as Dcpbst_827
except ImportError:
    pass

try:
    from .model_827_copy import Dcpbst as Dcpbst_827_Copy
except ImportError:
    pass

try:
    from .model_829_copy import Dcpbst as Dcpbst_829_Copy
except ImportError:
    pass

# Import configuration utilities
try:
    from .config_loader import load_config, create_model_from_config, get_available_configs
except ImportError:
    pass