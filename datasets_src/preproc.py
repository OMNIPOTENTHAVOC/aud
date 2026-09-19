"""
Preprocessing pipeline for ASVspoof2019 LA audio clips.

Every parameter here is pulled from configs/config.py rather than hardcoded,
so there is exactly one place (config.py) that defines the audio spec.
"""

import numpy as np
import librosa

from configs import config


def load_waveform(filepath):
    """Load a .flac clip at the configured sample rate."""
    waveform, _ = librosa.load(filepath, sr=config.SAMPLE_RATE)
    return waveform.astype(np.float32)


def trim_silence(waveform, top_db=25):
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed


def normalize_amplitude(waveform):
    peak = np.max(np.abs(waveform))
    return waveform / peak if peak > 0 else waveform


def pad_or_truncate(waveform, target_len=None):
    target_len = target_len or config.MAX_AUDIO_LENGTH
    if len(waveform) >= target_len:
        return waveform[:target_len]
    pad_width = target_len - len(waveform)
    return np.pad(waveform, (0, pad_width), mode="constant")


def extract_logmel_spectrogram(waveform):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        win_length=config.WIN_LENGTH,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # log compression
    return log_mel  # shape: (n_mels, time_frames)


def normalize_features(log_mel, mean=None, std=None):
    """Zero mean / unit variance.

    IMPORTANT: once you compute train-set mean/std (see compute_train_stats
    in dataset.py), pass them explicitly here for *every* split (train, dev,
    eval). Falling back to per-sample stats is only a placeholder for single
    ad-hoc clips -- using it during real training/eval would leak per-sample
    information and contradicts the train-set-only normalization decision.
    """
    if mean is None or std is None:
        mean, std = log_mel.mean(), log_mel.std()
    return (log_mel - mean) / (std + 1e-8)


def preprocess_clip(filepath, mean=None, std=None, normalize=True):
    """Full pipeline: load -> trim -> normalize amplitude -> pad/truncate -> log-mel -> normalize.

    Set normalize=False to get the raw log-mel values (before feature
    normalization) -- this is what compute_train_stats() in dataset.py uses,
    since computing "training statistics" on data that's already been
    force-normalized per clip would trivially yield mean~0, std~1 regardless
    of the actual data.
    """
    waveform = load_waveform(filepath)
    waveform = trim_silence(waveform)
    waveform = normalize_amplitude(waveform)
    waveform = pad_or_truncate(waveform)
    log_mel = extract_logmel_spectrogram(waveform)
    if normalize:
        log_mel = normalize_features(log_mel, mean=mean, std=std)
    return log_mel
