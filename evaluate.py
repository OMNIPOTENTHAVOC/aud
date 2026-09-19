"""
Evaluation entry point. Run from the project root, after training:

    python evaluate.py

Loads the best checkpoint, runs it on the eval set (unseen attacks
A07-A19), and reports overall metrics plus a per-attack-type EER
breakdown -- this breakdown is the secondary analysis layer for the
core contribution (paired later with the saliency-consistency correlation).
"""

import torch
from torch.utils.data import DataLoader

from configs import config
from datasets_src.dataset import ASVspoofDataset, compute_train_stats
from models.model import CNNLSTMAttention
from evaluation.inference import run_inference
from evaluation.metrics import compute_all_metrics, compute_per_attack_eer


def main(checkpoint_path="outputs/checkpoints/best_model.pt"):
    print(f"Device: {config.DEVICE}")

    # NOTE: normalization stats must be recomputed the same way as training
    # (train-set only) -- do not recompute stats from the eval set itself.
    mean, std = compute_train_stats()

    eval_ds = ASVspoofDataset("eval", mean=mean, std=std)
    eval_loader = DataLoader(eval_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"eval: {len(eval_ds)} clips (unseen attacks A07-A19)")

    model = CNNLSTMAttention(n_mels=config.N_MELS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    print(f"Loaded checkpoint: {checkpoint_path}")

    labels, scores, attack_ids = run_inference(model, eval_loader, config.DEVICE)

    print("\n=== Overall metrics (eval set) ===")
    overall = compute_all_metrics(labels, scores)
    for key, value in overall.items():
        print(f"  {key}: {value:.4f}")

    print("\n=== Per-attack-type EER breakdown ===")
    per_attack = compute_per_attack_eer(labels, scores, attack_ids)
    for attack_id, (eer, n_samples) in sorted(per_attack.items()):
        print(f"  {attack_id}: EER={eer:.4f}  (n={n_samples} spoof clips)")

    return overall, per_attack


if __name__ == "__main__":
    main()
