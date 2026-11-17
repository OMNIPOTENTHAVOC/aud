"""
Concentration (entropy) and consistency (SSIM) metrics on Grad-CAM heatmaps.

Entropy is the PRIMARY metric for the core contribution: low entropy =
tightly focused explanation, high entropy = scattered. Measured per
individual sample -- this sidesteps the "no ground-truth artifact location"
problem, since we're not claiming the heatmap points at the correct
location, only that it's confidently localized vs. diffuse.

SSIM-based consistency is SECONDARY: how similar are heatmaps across
samples of the same attack type.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_entropy(heatmap):
    """
    heatmap: 2D array (F', T'), non-negative (e.g. a Grad-CAM cam).
    Returns (raw_entropy, normalized_entropy) -- normalized is in [0, 1],
    dividing by log(N) so heatmaps of different sizes stay comparable.
    Lower = more concentrated/focused. Higher = more diffuse/scattered.
    """
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()
    flat = np.asarray(heatmap).flatten()
    flat = np.clip(flat, 1e-12, None)  # avoid log(0)
    p = flat / flat.sum()

    raw_entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(flat)))
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    return raw_entropy, normalized_entropy


def compute_consistency(heatmaps):
    """
    heatmaps: list of 2D arrays, all the same shape, from the SAME attack type.
    Returns mean pairwise SSIM -- higher = more consistent/similar
    explanations within this attack type. Returns None if fewer than 2
    heatmaps are given (can't compute a pairwise comparison).
    """
    heatmaps = [h.detach().cpu().numpy() if hasattr(h, "detach") else np.asarray(h) for h in heatmaps]
    if len(heatmaps) < 2:
        return None

    scores = []
    for i in range(len(heatmaps)):
        for j in range(i + 1, len(heatmaps)):
            a, b = heatmaps[i], heatmaps[j]
            data_range = max(a.max(), b.max()) - min(a.min(), b.min())
            data_range = data_range if data_range > 0 else 1.0
            scores.append(ssim(a, b, data_range=data_range))

    return float(np.mean(scores))
