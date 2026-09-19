"""
Reproducibility. Call set_seed() once, at the top of train.py, before
building the model or datasets -- without this, results (including which
epoch gets checkpointed as "best") aren't reproducible between runs.
"""

import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
