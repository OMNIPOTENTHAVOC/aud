"""
Main training entry point. Run from the project root:

    python train.py

Trains on the known-attack training set (A01-A06), validates each epoch on
dev, saves the best checkpoint by dev EER to outputs/checkpoints/best_model.pt.
"""

import json

import torch
from torch.utils.data import DataLoader

from configs import config
from datasets_src.dataset import ASVspoofDataset, compute_train_stats
from models.model import CNNLSTMAttention
from training.loss import compute_pos_weight, build_criterion
from training.optimizer import build_optimizer, build_scheduler
from training.train import train
from utils.seed import set_seed


def main():
    # Must run before any model/dataset construction -- both draw on the
    # global RNG state (weight init, DataLoader shuffling), so calling this
    # any later would leave those non-reproducible between runs.
    set_seed(42)

    print(f"Device: {config.DEVICE}")

    print("\nComputing training-set normalization statistics...")
    mean, std = compute_train_stats()
    print(f"mean={mean:.4f}, std={std:.4f}")

    print("\nBuilding datasets...")
    train_ds = ASVspoofDataset("train", mean=mean, std=std)
    dev_ds = ASVspoofDataset("dev", mean=mean, std=std)
    print(f"train: {len(train_ds)} clips, dev: {len(dev_ds)} clips")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = CNNLSTMAttention(n_mels=config.N_MELS)
    pos_weight = compute_pos_weight("train")
    criterion = build_criterion(pos_weight, config.DEVICE)
    optimizer = build_optimizer(model, config.LEARNING_RATE)
    scheduler = build_scheduler(optimizer, patience=3)

    checkpoint_path = "outputs/checkpoints/best_model.pt"
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print(f"Checkpoints will be saved to {checkpoint_path} whenever dev EER improves.\n")

    history, best_dev_eer = train(
        model, train_loader, dev_loader, criterion, optimizer, scheduler,
        device=config.DEVICE, num_epochs=config.NUM_EPOCHS,
        checkpoint_path=checkpoint_path,
    )

    print(f"\nTraining complete. Best dev EER: {best_dev_eer:.4f}")
    print(f"Best model saved to {checkpoint_path}")

    # Saved for downstream use -- scripts/generate_results_tex.py reads this
    # to report dataset sizes/hyperparameters and to plot the dev-EER
    # training curve for the paper.
    summary_path = config.PROJECT_ROOT / "outputs" / "predictions" / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "n_train_clips": len(train_ds),
        "n_dev_clips": len(dev_ds),
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "num_epochs_configured": config.NUM_EPOCHS,
        "num_epochs_run": len(history),
        "pos_weight": pos_weight,
        "best_dev_eer": best_dev_eer,
        "checkpoint_path": checkpoint_path,
        "history": history,
    }, indent=2))
    print(f"Saved training summary to {summary_path}")

    _plot_training_curve(history)


def _plot_training_curve(history):
    """Dev EER (model-selection criterion) vs. epoch -- the standard
    training-progress figure for the paper. Kept as a local helper (not a
    separate module) since it's only ever called from this entry point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    dev_eer = [h["eer"] for h in history]
    train_loss = [h["train_loss"] for h in history]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, dev_eer, color="tab:red", marker="o", label="Dev EER")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dev EER", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_loss, color="tab:blue", marker="s", alpha=0.6, label="Train loss")
    ax2.set_ylabel("Train loss", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig_dir = config.PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "training_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curve to {out_path}")


if __name__ == "__main__":
    main()
