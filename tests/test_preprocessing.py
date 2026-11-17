"""
Quick local smoke test -- confirms imports resolve, paths exist, and the
pipeline runs end to end on one real clip, before you commit to a full
training run. Run from the project root:

    python tests/test_preprocessing.py

If this fails on an import error, it's almost always the working-directory /
__init__.py issue described in the README, not a bug in the pipeline itself.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path even if this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config
from datasets_src.dataset import parse_protocol, SPLIT_DIRS
from datasets_src.preproc import preprocess_clip


def main():
    print(f"DEVICE: {config.DEVICE}")
    print(f"DATA_ROOT exists: {config.DATA_ROOT.exists()} -> {config.DATA_ROOT}")
    print(f"PROTOCOL_DIR exists: {config.PROTOCOL_DIR.exists()} -> {config.PROTOCOL_DIR}")

    entries = parse_protocol("train")
    print(f"\nParsed {len(entries)} entries from the train protocol.")
    assert len(entries) > 0, "Protocol parsed but found zero entries... check the file format."

    filename, attack_id, label = entries[0]
    filepath = SPLIT_DIRS["train"] / "flac" / f"{filename}.flac"
    print(f"\nTesting one real clip: {filepath}")
    assert filepath.exists(), f"Audio file not found at expected path: {filepath}"

    log_mel = preprocess_clip(filepath, normalize=False)
    print(f"Raw log-mel shape: {log_mel.shape}  (expect: (n_mels={config.N_MELS}, time_frames))")
    print(f"label={label}  attack_id={attack_id}")

    print("\nAll checks passed... pipeline wired correctly on machine.")


if __name__ == "__main__":
    main()
