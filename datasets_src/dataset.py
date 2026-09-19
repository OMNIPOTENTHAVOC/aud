"""
PyTorch Dataset for ASVspoof2019 LA.

Protocol file format (official ASVspoof2019 LA CM protocols), one line per clip:
    SPEAKER_ID  AUDIO_FILE_NAME  -  ATTACK_ID  KEY
e.g.
    LA_0079  LA_T_1138215  -  -    bonafide
    LA_0079  LA_T_1271820  -  A01  spoof

ATTACK_ID is '-' for bonafide clips, A01-A06 for the train/dev split, A07-A19
for the eval split (that seen/unseen split is baked into which protocol file
you load -- you don't need to filter by attack type yourself).

NOTE: the exact protocol filenames below (PROTOCOL_FILENAMES) match the
standard public release. If your PROTOCOL_DIR has different filenames,
adjust that dict -- resolve_protocol_file() will raise a clear error
listing what it actually found, rather than failing silently.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from configs import config
from datasets_src.preproc import preprocess_clip, normalize_features  # user's file: datasets_src/preproc.py

CACHE_DIR = config.PROJECT_ROOT / "outputs" / "spectrogram_cache"

PROTOCOL_FILENAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

SPLIT_DIRS = {
    "train": config.TRAIN_DIR,
    "dev": config.DEV_DIR,
    "eval": config.EVAL_DIR,
}

LABEL_MAP = {"bonafide": 0, "spoof": 1}  # 1 = spoof, matches BCEWithLogitsLoss target convention


def _atomic_save_npy(path, array):
    """np.save(path, array) writes directly to the final filename. If the
    process is killed mid-write (Ctrl+C, OOM kill, crash, power loss), the
    file at `path` is left truncated -- and since callers only check
    `path.exists()`, that corrupted file is trusted forever afterward and
    silently poisons every future run that hits it (this is exactly what
    produced "ValueError: Failed to read all data for array ... file seems
    not fully written?").

    Fix: write to a temporary file in the same directory, then rename it
    into place. Same-directory rename is atomic on POSIX filesystems, so a
    crash at any point leaves either no file or a fully-written one --
    never a partial one.
    """
    path = Path(path)
    # Temp file already ends in .npy, so np.save writes it byte-for-byte as
    # given rather than silently appending its own ".npy" suffix.
    tmp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npy")
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _load_cached_npy(path):
    """np.load(path), but treats a corrupted/truncated cache file (e.g. from
    a crash that happened before _atomic_save_npy was in place) as a cache
    miss instead of a fatal error: delete the bad file and return None so
    the caller recomputes and re-caches it. Self-heals old corrupted caches
    without needing a separate manual cleanup pass."""
    path = Path(path)
    try:
        return np.load(path)
    except (ValueError, EOFError, OSError) as e:
        print(f"[cache] Corrupted cache file, recomputing: {path} ({e})")
        path.unlink(missing_ok=True)
        return None


def resolve_protocol_file(split):
    expected = config.PROTOCOL_DIR / PROTOCOL_FILENAMES[split]
    if expected.exists():
        return expected
    found = list(config.PROTOCOL_DIR.glob(f"*{split}*"))
    raise FileNotFoundError(
        f"Expected protocol file not found: {expected}\n"
        f"Files matching '*{split}*' in {config.PROTOCOL_DIR}: {found}\n"
        f"Update PROTOCOL_FILENAMES in dataset.py if your filenames differ."
    )


def parse_protocol(split):
    """Returns a list of (filename_no_ext, attack_id, label_str) tuples."""
    protocol_path = resolve_protocol_file(split)
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed/blank lines rather than crash the whole load
            _speaker_id, filename, _env_id, attack_id, label = parts
            entries.append((filename, attack_id, label))
    return entries


def compute_train_stats(max_samples=None, force_recompute=False):
    """Compute log-mel mean/std across the training set (bona fide + spoof, known attacks).

    Run this once, then pass the resulting (mean, std) into ASVspoofDataset
    for train, dev, AND eval -- normalization stats must come from the
    training set only, never recomputed per split.

    Uses the same disk cache as ASVspoofDataset for per-clip raw values, and
    additionally caches the final (mean, std) itself to a small JSON file --
    so a second call (e.g. evaluate.py run right after train.py) loads two
    numbers instead of re-reading all ~25,380 training clips again.
    """
    stats_cache_path = config.PROJECT_ROOT / "outputs" / "train_stats.json"
    if not force_recompute and stats_cache_path.exists():
        cached = json.loads(stats_cache_path.read_text())
        return cached["mean"], cached["std"]

    entries = parse_protocol("train")
    if max_samples:
        entries = entries[:max_samples]

    cache_dir = CACHE_DIR / "train"
    cache_dir.mkdir(parents=True, exist_ok=True)

    running_sum, running_sq_sum, n_elements = 0.0, 0.0, 0
    flac_dir = SPLIT_DIRS["train"] / "flac"
    for filename, _attack_id, _label in entries:
        cache_path = cache_dir / f"{filename}.npy"
        log_mel = _load_cached_npy(cache_path) if cache_path.exists() else None
        if log_mel is None:
            filepath = flac_dir / f"{filename}.flac"
            log_mel = preprocess_clip(filepath, normalize=False)  # raw values -- see docstring above
            _atomic_save_npy(cache_path, log_mel)

        running_sum += log_mel.sum()
        running_sq_sum += (log_mel ** 2).sum()
        n_elements += log_mel.size

    mean = float(running_sum / n_elements)
    var = (running_sq_sum / n_elements) - (mean ** 2)
    std = float(np.sqrt(max(var, 1e-8)))

    stats_cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats_cache_path.write_text(json.dumps({"mean": mean, "std": std}))

    return mean, std


class ASVspoofDataset(Dataset):
    def __init__(self, split, mean=None, std=None, max_samples=None, use_cache=True):
        assert split in ("train", "dev", "eval"), f"Unknown split: {split}"
        self.split = split
        self.entries = parse_protocol(split)
        if max_samples:
            self.entries = self.entries[:max_samples]
        self.flac_dir = SPLIT_DIRS[split] / "flac"
        self.mean = mean
        self.std = std
        self.use_cache = use_cache

        if use_cache:
            self.cache_dir = CACHE_DIR / split
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.entries)

    def _get_raw_log_mel(self, filename, filepath):
        """Returns the RAW (pre-normalization) log-mel spectrogram, using the
        disk cache if available. Caching the raw values (not the normalized
        ones) means the cache stays valid even if mean/std ever changes."""
        if not self.use_cache:
            return preprocess_clip(filepath, normalize=False)

        cache_path = self.cache_dir / f"{filename}.npy"
        if cache_path.exists():
            log_mel = _load_cached_npy(cache_path)
            if log_mel is not None:
                return log_mel
            # _load_cached_npy already deleted the corrupted file; fall
            # through and recompute + re-cache it below.

        log_mel = preprocess_clip(filepath, normalize=False)
        _atomic_save_npy(cache_path, log_mel)
        return log_mel

    def __getitem__(self, idx):
        filename, attack_id, label = self.entries[idx]
        filepath = self.flac_dir / f"{filename}.flac"

        log_mel = self._get_raw_log_mel(filename, filepath)
        log_mel = normalize_features(log_mel, mean=self.mean, std=self.std)

        spec_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
        label_tensor = torch.tensor(LABEL_MAP[label], dtype=torch.float32)

        return spec_tensor, label_tensor, attack_id  # attack_id needed later for per-attack EER breakdown
