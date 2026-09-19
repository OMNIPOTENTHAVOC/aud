"""
Plotting helpers for the dual-channel interpretability output:
spectral (Grad-CAM) + temporal (attention weights).
"""

import matplotlib
matplotlib.use("Agg")  # safe for headless/server environments; drop this line if you want interactive plots
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_gradcam_overlay(spectrogram, gradcam_map, title="Grad-CAM overlay", save_path=None):
    """
    spectrogram: 2D array (n_mels, time_frames) -- the full-resolution log-mel spectrogram
    gradcam_map: 2D array (F', T') -- lower-resolution Grad-CAM output, gets upsampled to match
    """
    if hasattr(spectrogram, "detach"):
        spectrogram = spectrogram.detach().cpu().numpy()
    if hasattr(gradcam_map, "detach"):
        gradcam_map = gradcam_map.detach().cpu().numpy()

    gradcam_t = torch.tensor(gradcam_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(gradcam_t, size=spectrogram.shape, mode="bilinear", align_corners=False)
    upsampled = upsampled.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Log-mel spectrogram")
    axes[0].set_xlabel("time frame")
    axes[0].set_ylabel("mel bin")

    axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="gray")
    axes[1].imshow(upsampled, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[1].set_title(title)
    axes[1].set_xlabel("time frame")
    axes[1].set_ylabel("mel bin")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_weights(weights, title="Attention weights (temporal channel)", save_path=None):
    """weights: 1D array (T',) -- softmax attention weights over reduced time steps."""
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights).squeeze()

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(range(len(weights)), weights, color="steelblue")
    ax.set_xlabel("reduced time step")
    ax.set_ylabel("attention weight")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_combined(spectrogram, gradcam_map, attention_weights, save_path=None):
    """One figure: spectrogram, Grad-CAM overlay, and attention weights side by side."""
    if hasattr(spectrogram, "detach"):
        spectrogram = spectrogram.detach().cpu().numpy()
    if hasattr(gradcam_map, "detach"):
        gradcam_map = gradcam_map.detach().cpu().numpy()
    if hasattr(attention_weights, "detach"):
        attention_weights = attention_weights.detach().cpu().numpy()
    attention_weights = np.asarray(attention_weights).squeeze()

    gradcam_t = torch.tensor(gradcam_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(gradcam_t, size=spectrogram.shape, mode="bilinear", align_corners=False)
    upsampled = upsampled.squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Spectrogram")

    axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="gray")
    axes[1].imshow(upsampled, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[1].set_title("Grad-CAM (spectral)")

    axes[2].bar(range(len(attention_weights)), attention_weights, color="steelblue")
    axes[2].set_title("Attention (temporal)")
    axes[2].set_xlabel("reduced time step")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
