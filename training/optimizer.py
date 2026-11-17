"""
Optimizer + LR scheduler.

Adds a ReduceLROnPlateau scheduler tied to dev-set EER (not train loss) --
config.py's fixed 30-epoch budget had no scheduler, so a lucky/unlucky
epoch 30 would otherwise decide the reported number. This backs off the
learning rate automatically if dev EER stalls.
"""

import torch


def build_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def build_scheduler(optimizer, patience=3, factor=0.5):
    # mode="min" because we're tracking dev EER -- lower is better
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )
