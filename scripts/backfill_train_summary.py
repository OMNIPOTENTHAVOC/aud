"""
Reconstructs outputs/predictions/train_summary.json from an ALREADY-TRAINED
checkpoint, without retraining -- for the case where training finished
before train.py had the JSON-saving code added, and no console log was
kept.

What this CAN recover without retraining, and why:
  - best_dev_eer: the saved checkpoint is, by construction, the one with
    the lowest dev EER seen during training (train.py only overwrites it
    on improvement) -- so running that exact checkpoint against the dev
    set once and computing EER reproduces best_dev_eer exactly.
  - n_train_clips / n_dev_clips: dataset sizes are just protocol-file
    lengths (datasets_src.dataset.parse_protocol) -- no audio decoding
    needed, so this is fast regardless of dataset size.
  - batch_size / learning_rate / num_epochs_configured: read directly
    from configs/config.py.
  - pos_weight: training.loss.compute_pos_weight only counts protocol
    labels -- also fast, no retraining involved.

What this CANNOT recover, and why a retrain (or a kept log) is the only
way to get it:
  - num_epochs_run and the full per-epoch history (train_loss/dev_eer per
    epoch): these describe the *trajectory* across training, which only
    exists if it was recorded while training ran. A finished run's final
    checkpoint has no memory of that trajectory. These fields are left out
    of the written JSON entirely (not set to "TODO" -- see
    scripts/generate_results_tex.py, which falls back to "TODO" for any
    missing key), and the training-curve figure will stay empty until you
    either retrain with the current train.py or supply a saved log.

Run from the project root:

    python scripts/backfill_train_summary.py
    python scripts/backfill_train_summary.py --checkpoint outputs/checkpoints/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config
from datasets_src.dataset import ASVspoofDataset, compute_train_stats, parse_protocol
from models.model import CNNLSTMAttention
from evaluation.inference import run_inference
from evaluation.metrics import compute_all_metrics
from training.loss import compute_pos_weight


def main(checkpoint_path="outputs/checkpoints/best_model.pt"):
    print(f"Device: {config.DEVICE}")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- this script recovers "
            f"train_summary.json from an existing checkpoint, so training "
            f"must have actually produced one first."
        )

    print("Recomputing dev EER from the saved checkpoint (this is exactly "
          "best_dev_eer, by construction -- see module docstring) ...")
    mean, std = compute_train_stats()
    dev_ds = ASVspoofDataset("dev", mean=mean, std=std)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = CNNLSTMAttention(n_mels=config.N_MELS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.to(config.DEVICE)

    labels, scores, _attack_ids = run_inference(model, dev_loader, config.DEVICE)
    dev_metrics = compute_all_metrics(labels, scores)
    best_dev_eer = dev_metrics["eer"]
    print(f"  Recomputed best_dev_eer = {best_dev_eer:.4f}")

    print("Counting dataset sizes and recomputing pos_weight "
          "(protocol-file only, no audio decoding) ...")
    n_train_clips = len(parse_protocol("train"))
    n_dev_clips = len(dev_ds)
    pos_weight = compute_pos_weight("train")

    summary = {
        "n_train_clips": n_train_clips,
        "n_dev_clips": n_dev_clips,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "num_epochs_configured": config.NUM_EPOCHS,
        "pos_weight": pos_weight,
        "best_dev_eer": best_dev_eer,
        "checkpoint_path": str(checkpoint_path),
        # num_epochs_run and history deliberately omitted -- see docstring.
        "note": (
            "Backfilled from an already-trained checkpoint by "
            "scripts/backfill_train_summary.py, not from train.py's own "
            "epoch loop. num_epochs_run and per-epoch history are "
            "unavailable, so the training-curve figure will stay empty "
            "until a retrain (with the current train.py) or a saved log "
            "is available."
        ),
    }

    summary_path = config.PROJECT_ROOT / "outputs" / "predictions" / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved backfilled training summary to {summary_path}")
    print("NOTE: no training-curve figure data -- NumEpochsRun will stay "
          "\"TODO\" in the paper unless you retrain or supply a log.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint)
