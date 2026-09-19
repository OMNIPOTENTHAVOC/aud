"""
Interpretability driver -- ties gradcam.py + entropy.py + visualize.py
together into the actual per-attack-type results: this is the piece
evaluate.py's docstring refers to as "paired later with the
saliency-consistency correlation," and nothing produced it until now.

For each attack type in the given split:
  1. Runs Grad-CAM on a capped, seeded random subsample of clips
     (full eval set is tens of thousands of clips x one backward pass
     each -- not worth the wall-clock for a same-day result).
  2. Computes normalized entropy per clip (concentration -- PRIMARY metric).
  3. Computes mean pairwise SSIM consistency across that attack type's
     heatmaps (SECONDARY metric).
  4. Saves a few example Grad-CAM + attention overlay figures per attack.

Also runs the fragility noise-floor control ONCE across a random sample
spanning all attack types, so low per-attack consistency can be read
against "Grad-CAM is just fragile in general" rather than assumed to be
a real finding.

Run from the project root, after training and evaluate.py:

    python interpretability/run_interpretability.py
    python interpretability/run_interpretability.py --split dev --checkpoint outputs/checkpoints/best_model.pt

Outputs:
    outputs/predictions/interpretability_summary.json
    outputs/figures/gradcam_<attack_id>_<i>.png
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from configs import config
from datasets_src.dataset import ASVspoofDataset, compute_train_stats
from models.model import CNNLSTMAttention
from interpretability.gradcam import compute_gradcam, compute_fragility
from interpretability.entropy import compute_entropy, compute_consistency
from interpretability.visualize import plot_combined

N_EXAMPLE_FIGURES_PER_ATTACK = 3
N_FRAGILITY_SAMPLES = 10
MAX_CLIPS_PER_ATTACK = 40  # subsample cap -- see module docstring
SEED = 42


def load_model(checkpoint_path, device):
    model = CNNLSTMAttention(n_mels=config.N_MELS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def group_indices_by_attack(dataset, max_per_attack=MAX_CLIPS_PER_ATTACK, seed=SEED):
    """{attack_id: [dataset indices]}, spoof clips only, capped and shuffled
    deterministically so results are reproducible run-to-run."""
    by_attack = defaultdict(list)
    for idx, (_filename, attack_id, _label) in enumerate(dataset.entries):
        if attack_id != "-":
            by_attack[attack_id].append(idx)

    rng = random.Random(seed)
    for attack_id in by_attack:
        rng.shuffle(by_attack[attack_id])
        by_attack[attack_id] = by_attack[attack_id][:max_per_attack]
    return by_attack


def run_fragility_control(model, dataset, all_indices, device, n_samples=N_FRAGILITY_SAMPLES, seed=SEED):
    rng = random.Random(seed)
    sample_indices = rng.sample(all_indices, min(n_samples, len(all_indices)))

    shifts = []
    for idx in sample_indices:
        spec, _label, _attack_id = dataset[idx]
        spec = spec.unsqueeze(0).to(device)
        result = compute_fragility(model, spec, n_trials=5, noise_std=0.01)
        shifts.append(result["mean_shift"])

    return {
        "mean_shift": float(sum(shifts) / len(shifts)),
        "n_clips": len(sample_indices),
    }


def main(checkpoint_path="outputs/checkpoints/best_model.pt", split="eval"):
    device = config.DEVICE
    print(f"Device: {device}")

    mean, std = compute_train_stats()
    ds = ASVspoofDataset(split, mean=mean, std=std)
    print(f"{split}: {len(ds)} clips")

    model = load_model(checkpoint_path, device)

    by_attack = group_indices_by_attack(ds)
    print(f"Attack types found: {sorted(by_attack.keys())}")

    fig_dir = config.PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for attack_id, indices in sorted(by_attack.items()):
        print(f"\n--- {attack_id} ({len(indices)} clips) ---")
        entropies = []
        heatmaps = []

        for i, idx in enumerate(indices):
            spec, _label, _aid = ds[idx]
            spec = spec.unsqueeze(0).to(device)  # (1, 1, n_mels, T)

            cam = compute_gradcam(model, spec)  # (1, F', T')
            cam_2d = cam[0].cpu().numpy()
            _raw_entropy, norm_entropy = compute_entropy(cam_2d)
            entropies.append(norm_entropy)
            heatmaps.append(cam_2d)

            if i < N_EXAMPLE_FIGURES_PER_ATTACK:
                with torch.no_grad():
                    _logits, attn = model(spec, return_attention=True)
                plot_combined(
                    spectrogram=spec[0, 0].cpu().numpy(),
                    gradcam_map=cam_2d,
                    attention_weights=attn[0].cpu().numpy(),
                    save_path=fig_dir / f"gradcam_{attack_id}_{i}.png",
                )

        consistency = compute_consistency(heatmaps)
        mean_entropy = sum(entropies) / len(entropies)
        std_entropy = (
            float(torch.tensor(entropies).std().item()) if len(entropies) > 1 else 0.0
        )

        summary[attack_id] = {
            "n_clips": len(indices),
            "mean_entropy": float(mean_entropy),
            "std_entropy": std_entropy,
            "consistency_ssim": consistency,
        }

        consistency_str = f"{consistency:.4f}" if consistency is not None else "N/A"
        print(f"  mean_entropy={mean_entropy:.4f}  consistency_ssim={consistency_str}")

    print("\n--- Fragility noise-floor control ---")
    all_indices = [idx for indices in by_attack.values() for idx in indices]
    fragility = run_fragility_control(model, ds, all_indices, device)
    print(f"  mean pixelwise CAM shift under noise alone: {fragility['mean_shift']:.4f} "
          f"(n={fragility['n_clips']} clips)")
    summary["_fragility_noise_floor"] = fragility

    out_path = config.PROJECT_ROOT / "outputs" / "predictions" / "interpretability_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_path}")
    print(f"Saved example overlay figures to {fig_dir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--split", default="eval", choices=["dev", "eval"])
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint, split=args.split)
