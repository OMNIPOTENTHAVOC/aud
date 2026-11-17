"""
Model smoke test -- confirms the architecture builds correctly and a
full batch flows through it (dummy data), then confirms the model works
against one real preprocessed clip from the actual dataset. Run from the
project root:

    python tests/test_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.model import CNNLSTMAttention
from datasets_src.dataset import ASVspoofDataset, compute_train_stats
from configs import config


def test_dummy_batch():
    model = CNNLSTMAttention()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built. Total parameters: {n_params:,}")

    dummy = torch.randn(4, 1, config.N_MELS, 400)
    logits, attn = model(dummy, return_attention=True)

    assert logits.shape == (4,), f"Unexpected logits shape: {logits.shape}"
    assert attn.shape[0] == 4, f"Unexpected attention batch dim: {attn.shape}"
    assert model.last_conv_features is not None, "last_conv_features not populated -- Grad-CAM won't work"

    print(f"Dummy batch OK: logits={tuple(logits.shape)}, attention={tuple(attn.shape)}")


def test_real_clip():
    mean, std = compute_train_stats()
    ds = ASVspoofDataset("train", mean=mean, std=std)

    model = CNNLSTMAttention()
    model.eval()

    spec, label, attack_id = ds[0]
    spec = spec.unsqueeze(0)  # add batch dim

    with torch.no_grad():
        logits, attn = model(spec, return_attention=True)
        prob_spoof = torch.sigmoid(logits).item()

    print(f"Real clip OK: label={label.item()}, attack_id={attack_id}, "
          f"P(spoof)={prob_spoof:.4f} (untrained weights -- not meaningful yet)")


if __name__ == "__main__":
    test_dummy_batch()
    print()
    test_real_clip()
    print("\nAll model checks passed.")
