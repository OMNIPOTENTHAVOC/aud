"""
Secondary metrics: F1 and AUC. EER (in eer.py) remains the primary metric.
"""

from sklearn.metrics import f1_score, roc_auc_score

from evaluation.eer import compute_eer


def compute_f1(y_true, y_score, threshold=0.5):
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return f1_score(y_true, y_pred)


def compute_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def compute_per_attack_eer(y_true, y_score, attack_ids):
    """
    Standard ASVspoof per-attack evaluation: for each attack type, compute
    EER using ALL bonafide clips plus only that attack type's spoof clips
    (a single attack type alone has no negative examples to compare against).

    Returns {attack_id: (eer, n_spoof_samples)}.
    """
    results = {}
    unique_attacks = sorted(set(a for a in attack_ids if a != "-"))

    for attack in unique_attacks:
        mask = [(a == attack or a == "-") for a in attack_ids]
        subset_labels = [l for l, m in zip(y_true, mask) if m]
        subset_scores = [s for s, m in zip(y_score, mask) if m]
        n_spoof = sum(1 for a in attack_ids if a == attack)

        eer, _threshold = compute_eer(subset_labels, subset_scores)
        results[attack] = (eer, n_spoof)

    return results


def compute_all_metrics(y_true, y_score):
    eer, eer_threshold = compute_eer(y_true, y_score)
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        # F1 at a fixed 0.5 threshold can look artificially bad early in
        # training, purely because the model's outputs haven't spread out
        # from ~0.5 yet -- NOT because ranking is bad. F1 at the EER
        # threshold gives a fairer read on classification quality itself.
        "f1_at_0.5": compute_f1(y_true, y_score, threshold=0.5),
        "f1_at_eer_threshold": compute_f1(y_true, y_score, threshold=eer_threshold),
        "auc": compute_auc(y_true, y_score),
    }
