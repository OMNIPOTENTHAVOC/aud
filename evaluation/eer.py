"""
Equal Error Rate (EER) -- the standard metric in anti-spoofing literature.
The threshold where false positive rate equals false negative rate.
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true, y_score):
    """
    y_true: 0/1 labels (1 = spoof)
    y_score: predicted probability of spoof (higher = more likely spoof)

    Returns (eer, eer_threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    # index where FPR and FNR are closest -- the EER crossing point
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return float(eer), float(eer_threshold)
