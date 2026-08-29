# Legacy Model Files

These model files are kept for reference only. They contain dataset-specific configurations
that have been consolidated into the unified model at `dcpbst_package/model.py`.

## How to use

For new experiments, use the unified model:
```python
from dcpbst_package.model import Dcpbst
from dcpbst_package.configs import load_config
```

These legacy files are maintained for backward compatibility with existing notebooks.

## Model variants

- `model_827.py` — Base model for DLPFC, BRAC, Mouse AREI (Visium)
- `model_827_copy.py` — PDAC variant (different hyperparameters, POT import)
- `model_829_copy.py` — MERFISH/Hypothalamus variant (no image features, different label handling)
