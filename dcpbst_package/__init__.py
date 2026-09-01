"""
DCPBST: Multi-modal Integration for Spatial Omics.

Core package providing the DCPBST model variants used by the reproducibility
notebooks and scripts. Each dataset imports the variant it was developed with:

  - model_827_copy.py   : DLPFC (151509 / 151671), BRCA, PDAC and Mouse AREI
  - model_829_copy.py   : Mouse Hypothalamus (MERFISH)
"""

try:
    from .model_827_copy import Dcpbst as Dcpbst_827_Copy
except ImportError:
    pass

try:
    from .model_829_copy import Dcpbst as Dcpbst_829_Copy
except ImportError:
    pass
