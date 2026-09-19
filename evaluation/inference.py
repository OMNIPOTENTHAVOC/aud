"""
Runs a model over a DataLoader and collects predictions, labels, and
attack_ids -- the attack_ids are what later enables the per-attack-type
EER breakdown, not just an overall number.
"""

import torch


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    all_labels, all_scores, all_attack_ids = [], [], []

    for specs, labels, attack_ids in dataloader:
        specs = specs.to(device)
        logits = model(specs)
        scores = torch.sigmoid(logits).cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(scores.tolist())
        all_attack_ids.extend(attack_ids)

    return all_labels, all_scores, all_attack_ids
