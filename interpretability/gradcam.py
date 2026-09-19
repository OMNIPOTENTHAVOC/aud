"""
Grad-CAM (spectral channel) attached to the last CNN layer only -- not
extended through the LSTM, per the locked design decision: standard
Grad-CAM assumes the attributed layer is convolutional, and there's no
settled convention for extending it through a recurrent layer. The
temporal question is answered separately by attention weights.

Also includes the fragility-control baseline: measures how much a Grad-CAM
heatmap shifts under tiny, inaudible input noise alone. Without this,
"attack type X has an inconsistent explanation" can't be told apart from
"Grad-CAM is just fragile in general" -- a well-documented property of
gradient-based saliency methods.
"""

import torch


def compute_gradcam(model, spec_tensor):
    """
    model: CNNLSTMAttention, in eval() mode
    spec_tensor: (batch, 1, n_mels, time_frames), requires no grad itself

    Returns: cam, shape (batch, F', T') -- normalized to [0, 1] per sample.
    """
    model.eval()
    spec_tensor = spec_tensor.clone()

    logits = model(spec_tensor)              # forward pass populates model.last_conv_features
    feats = model.last_conv_features
    feats.retain_grad()                       # need gradients w.r.t. this intermediate tensor

    model.zero_grad()
    logits.sum().backward()                   # safe per-sample in eval() mode: BatchNorm uses
                                                # running stats, not batch stats, so no cross-sample leakage

    grads = feats.grad                         # (batch, C, F', T')
    activations = feats.detach()               # (batch, C, F', T')

    weights = grads.mean(dim=(2, 3), keepdim=True)         # GAP gradients per channel
    cam = torch.relu((weights * activations).sum(dim=1))    # weighted sum -> ReLU, (batch, F', T')

    # normalize each sample in the batch independently to [0, 1]
    batch = cam.shape[0]
    cam_flat = cam.view(batch, -1)
    cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(batch, 1, 1)
    cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(batch, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    return cam.detach()


def compute_fragility(model, spec_tensor, n_trials=5, noise_std=0.01):
    """
    Baseline for Grad-CAM's own sensitivity to imperceptible input noise.
    Run this BEFORE trusting any "this attack type's explanation is
    inconsistent" claim -- compare real cross-sample variation against
    this noise floor, not against zero.

    spec_tensor: single sample, (1, 1, n_mels, time_frames)
    Returns dict with mean/std pixel-wise shift across n_trials tiny
    perturbations, plus the baseline (unperturbed) cam for reference.
    """
    baseline_cam = compute_gradcam(model, spec_tensor)

    shifts = []
    for _ in range(n_trials):
        noisy = spec_tensor + torch.randn_like(spec_tensor) * noise_std
        noisy_cam = compute_gradcam(model, noisy)
        shift = torch.mean(torch.abs(baseline_cam - noisy_cam)).item()
        shifts.append(shift)

    return {
        "baseline_cam": baseline_cam,
        "mean_shift": float(sum(shifts) / len(shifts)),
        "std_shift": float(torch.tensor(shifts).std().item()) if len(shifts) > 1 else 0.0,
        "n_trials": n_trials,
        "noise_std": noise_std,
    }
