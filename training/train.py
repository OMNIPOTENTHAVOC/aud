"""
Training loop. Tracks dev-set EER (not train loss) as the model-selection
criterion, since EER is the metric that actually matters for this task and
train loss can look good while generalization doesn't.
"""

from pathlib import Path

import torch

from evaluation.inference import run_inference
from evaluation.metrics import compute_all_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for specs, labels, _attack_ids in dataloader:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(model, train_loader, dev_loader, criterion, optimizer, scheduler,
          device, num_epochs, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    best_dev_eer = float("inf")
    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        dev_labels, dev_scores, _dev_attack_ids = run_inference(model, dev_loader, device)
        dev_metrics = compute_all_metrics(dev_labels, dev_scores)

        scheduler.step(dev_metrics["eer"])

        improved = dev_metrics["eer"] < best_dev_eer
        if improved:
            best_dev_eer = dev_metrics["eer"]
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.4f} | "
              f"dev_eer={dev_metrics['eer']:.4f} | dev_f1@eer_thresh={dev_metrics['f1_at_eer_threshold']:.4f} | "
              f"dev_auc={dev_metrics['auc']:.4f} {'(saved)' if improved else ''}")

        history.append({"epoch": epoch, "train_loss": train_loss, **dev_metrics})

    return history, best_dev_eer
