"""
Main training entry point. Run from the project root:

    python train.py

Trains on the known-attack training set (A01-A06), validates each epoch on
dev, saves the best checkpoint by dev EER to outputs/checkpoints/best_model.pt.
"""

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


if __name__ == "__main__":
    main()
