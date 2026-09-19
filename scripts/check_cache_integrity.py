"""
Scans outputs/spectrogram_cache/ for corrupted (truncated) .npy files --
the kind left behind by a process that was killed mid-write before
datasets_src/dataset.py's atomic-write fix was in place -- and reports or
removes them.

This is a one-time cleanup for cache files written before the fix. After
the fix, dataset.py self-heals any corrupted file it happens to hit during
normal use, so this script is optional going forward; it is mainly useful
to check for damage up front instead of finding out one crash at a time.

Run from the project root:

    python scripts/check_cache_integrity.py          # report only
    python scripts/check_cache_integrity.py --delete  # also delete bad files
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# scripts/ is not the project root -- python only puts this file's own
# directory on sys.path by default, so "from configs import config" would
# fail (ModuleNotFoundError: No module named 'configs') without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config

CACHE_DIR = config.PROJECT_ROOT / "outputs" / "spectrogram_cache"


def main(delete=False):
    if not CACHE_DIR.exists():
        print(f"No cache directory at {CACHE_DIR} -- nothing to check.")
        return

    npy_files = sorted(CACHE_DIR.rglob("*.npy"))
    print(f"Checking {len(npy_files)} cached files under {CACHE_DIR} ...")

    bad = []
    for path in npy_files:
        try:
            np.load(path)
        except (ValueError, EOFError, OSError) as e:
            bad.append((path, str(e)))

    if not bad:
        print("No corrupted files found.")
        return

    print(f"\nFound {len(bad)} corrupted file(s):")
    for path, err in bad:
        print(f"  {path}\n    -> {err}")

    if delete:
        for path, _err in bad:
            path.unlink()
        print(f"\nDeleted {len(bad)} corrupted file(s). They will be "
              f"recomputed automatically the next time they're accessed.")
    else:
        print(f"\nRun with --delete to remove these {len(bad)} file(s). "
              f"They will be recomputed automatically the next time they're "
              f"accessed either way (dataset.py now self-heals on read), "
              f"so this is just for visibility into how many were affected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true",
                         help="Delete corrupted files instead of only reporting them.")
    args = parser.parse_args()
    main(delete=args.delete)
