"""
Unit tests for datasets_src/dataset.py.

IMPORTANT GOTCHA this test file works around: CACHE_DIR, PROTOCOL_FILENAMES,
and SPLIT_DIRS in dataset.py are computed ONCE at import time from
config.PROJECT_ROOT / config.TRAIN_DIR / etc. Monkeypatching config.* after
import does NOT change them -- you have to monkeypatch dataset.CACHE_DIR /
dataset.SPLIT_DIRS directly. (config.PROTOCOL_DIR is the one exception:
resolve_protocol_file() re-reads it from the config module on every call,
so patching config.PROTOCOL_DIR does work for protocol-parsing tests.)
This is worth fixing in the source (read paths lazily / pass them in) if
these globals ever need to point somewhere else at runtime, e.g. in tests
or a multi-dataset-root setup.

Run from the project root:

    pytest tests/test_dataset_unit.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config
from datasets_src import dataset


PROTOCOL_LINE = "{speaker} {filename} - {attack} {label}\n"


def write_protocol(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# resolve_protocol_file / parse_protocol
# ---------------------------------------------------------------------------

class TestResolveProtocolFile:
    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROTOCOL_DIR", tmp_path)
        expected = tmp_path / dataset.PROTOCOL_FILENAMES["train"]
        expected.write_text("")
        assert dataset.resolve_protocol_file("train") == expected

    def test_raises_with_helpful_message_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROTOCOL_DIR", tmp_path)
        # a similarly-named file exists but not the exact expected name --
        # the error should surface it as a hint, not just say "not found"
        decoy = tmp_path / "some_other_train_file.txt"
        decoy.write_text("")

        with pytest.raises(FileNotFoundError) as exc_info:
            dataset.resolve_protocol_file("train")

        message = str(exc_info.value)
        assert "train" in message
        assert str(decoy) in message or decoy.name in message


class TestParseProtocol:
    def test_parses_well_formed_lines(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROTOCOL_DIR", tmp_path)
        protocol_path = tmp_path / dataset.PROTOCOL_FILENAMES["train"]
        write_protocol(protocol_path, [
            PROTOCOL_LINE.format(speaker="LA_0079", filename="LA_T_0001", attack="-", label="bonafide"),
            PROTOCOL_LINE.format(speaker="LA_0079", filename="LA_T_0002", attack="A01", label="spoof"),
        ])

        entries = dataset.parse_protocol("train")

        assert entries == [
            ("LA_T_0001", "-", "bonafide"),
            ("LA_T_0002", "A01", "spoof"),
        ]

    def test_skips_malformed_lines_instead_of_crashing(self, tmp_path, monkeypatch):
        """Per the module docstring, malformed/blank lines should be
        skipped rather than blowing up the entire load."""
        monkeypatch.setattr(config, "PROTOCOL_DIR", tmp_path)
        protocol_path = tmp_path / dataset.PROTOCOL_FILENAMES["dev"]
        write_protocol(protocol_path, [
            "too few fields\n",
            "\n",  # blank line
            PROTOCOL_LINE.format(speaker="LA_0080", filename="LA_D_0001", attack="-", label="bonafide"),
            "way too many fields here for sure yes\n",
        ])

        entries = dataset.parse_protocol("dev")

        assert entries == [("LA_D_0001", "-", "bonafide")]

    def test_empty_protocol_file_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROTOCOL_DIR", tmp_path)
        (tmp_path / dataset.PROTOCOL_FILENAMES["eval"]).write_text("")
        assert dataset.parse_protocol("eval") == []


# ---------------------------------------------------------------------------
# compute_train_stats
# ---------------------------------------------------------------------------

class TestComputeTrainStats:
    def test_uses_cached_stats_without_recomputing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        stats_path = tmp_path / "outputs" / "train_stats.json"
        stats_path.parent.mkdir(parents=True)
        stats_path.write_text(json.dumps({"mean": 1.23, "std": 4.56}))

        # If parse_protocol gets called at all, the cache path wasn't
        # actually short-circuiting -- fail loudly instead of silently
        # doing real (slow, file-dependent) work.
        def fail_if_called(split):
            raise AssertionError("parse_protocol should not run when a valid cache exists")

        monkeypatch.setattr(dataset, "parse_protocol", fail_if_called)

        mean, std = dataset.compute_train_stats()
        assert (mean, std) == (1.23, 4.56)

    def test_force_recompute_ignores_existing_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        cache_dir = tmp_path / "spectrogram_cache"
        monkeypatch.setattr(dataset, "CACHE_DIR", cache_dir)

        stats_path = tmp_path / "outputs" / "train_stats.json"
        stats_path.parent.mkdir(parents=True)
        stats_path.write_text(json.dumps({"mean": -999.0, "std": -999.0}))

        monkeypatch.setattr(dataset, "parse_protocol", lambda split: [("clipA", "-", "bonafide")])
        monkeypatch.setattr(
            dataset, "preprocess_clip",
            lambda filepath, normalize: np.array([[2.0, 4.0], [6.0, 8.0]]),
        )
        monkeypatch.setattr(dataset, "SPLIT_DIRS", {**dataset.SPLIT_DIRS, "train": tmp_path / "train"})

        mean, std = dataset.compute_train_stats(force_recompute=True)

        assert mean == pytest.approx(5.0)  # mean of [2,4,6,8]
        # population std of [2,4,6,8]: var = mean((x-5)^2) = (9+1+1+9)/4 = 5
        assert std == pytest.approx(np.sqrt(5.0), rel=1e-4)

    def test_computes_correct_mean_std_across_multiple_clips(self, tmp_path, monkeypatch):
        """Verifies the running-sum accumulation is correct across >1 clip,
        not just correct for a single array (that's the case most likely
        to hide an accumulation bug)."""
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dataset, "CACHE_DIR", tmp_path / "spectrogram_cache")
        monkeypatch.setattr(dataset, "SPLIT_DIRS", {**dataset.SPLIT_DIRS, "train": tmp_path / "train"})
        monkeypatch.setattr(
            dataset, "parse_protocol",
            lambda split: [("clipA", "-", "bonafide"), ("clipB", "A01", "spoof")],
        )

        clips = {
            "clipA": np.array([[0.0, 0.0]]),
            "clipB": np.array([[10.0, 10.0]]),
        }
        monkeypatch.setattr(
            dataset, "preprocess_clip",
            lambda filepath, normalize: clips[filepath.stem],
        )

        mean, std = dataset.compute_train_stats(force_recompute=True)

        all_values = np.array([0.0, 0.0, 10.0, 10.0])
        assert mean == pytest.approx(all_values.mean())
        assert std == pytest.approx(all_values.std(), rel=1e-4)

    def test_respects_max_samples(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dataset, "CACHE_DIR", tmp_path / "spectrogram_cache")
        monkeypatch.setattr(dataset, "SPLIT_DIRS", {**dataset.SPLIT_DIRS, "train": tmp_path / "train"})
        monkeypatch.setattr(
            dataset, "parse_protocol",
            lambda split: [(f"clip{i}", "-", "bonafide") for i in range(100)],
        )
        calls = []
        monkeypatch.setattr(
            dataset, "preprocess_clip",
            lambda filepath, normalize: calls.append(filepath.stem) or np.array([[1.0]]),
        )

        dataset.compute_train_stats(max_samples=3, force_recompute=True)

        assert len(calls) == 3


# ---------------------------------------------------------------------------
# ASVspoofDataset
# ---------------------------------------------------------------------------

class TestASVspoofDataset:
    def _build(self, tmp_path, monkeypatch, entries, use_cache=True, max_samples=None):
        monkeypatch.setattr(dataset, "parse_protocol", lambda split: entries)
        monkeypatch.setattr(dataset, "CACHE_DIR", tmp_path / "spectrogram_cache")
        monkeypatch.setattr(dataset, "SPLIT_DIRS", {**dataset.SPLIT_DIRS, "train": tmp_path / "train"})
        return dataset.ASVspoofDataset("train", mean=0.0, std=1.0, use_cache=use_cache, max_samples=max_samples)

    def test_rejects_unknown_split(self):
        with pytest.raises(AssertionError):
            dataset.ASVspoofDataset("bogus_split")

    def test_len_reflects_max_samples(self, tmp_path, monkeypatch):
        entries = [(f"clip{i}", "-", "bonafide") for i in range(10)]
        ds = self._build(tmp_path, monkeypatch, entries, max_samples=4)
        assert len(ds) == 4

    def test_getitem_bonafide_maps_to_label_zero(self, tmp_path, monkeypatch):
        import torch

        ds = self._build(tmp_path, monkeypatch, [("clipA", "-", "bonafide")])
        monkeypatch.setattr(dataset, "preprocess_clip", lambda filepath, normalize: np.ones((4, 4)))
        monkeypatch.setattr(dataset, "normalize_features", lambda log_mel, mean, std: log_mel)

        spec_tensor, label_tensor, attack_id = ds[0]

        assert label_tensor.item() == 0.0
        assert attack_id == "-"
        assert spec_tensor.shape == (1, 4, 4)  # channel dim unsqueezed
        assert spec_tensor.dtype == torch.float32

    def test_getitem_spoof_maps_to_label_one(self, tmp_path, monkeypatch):
        ds = self._build(tmp_path, monkeypatch, [("clipB", "A01", "spoof")])
        monkeypatch.setattr(dataset, "preprocess_clip", lambda filepath, normalize: np.ones((4, 4)))
        monkeypatch.setattr(dataset, "normalize_features", lambda log_mel, mean, std: log_mel)

        _spec, label_tensor, attack_id = ds[0]

        assert label_tensor.item() == 1.0
        assert attack_id == "A01"

    def test_disk_cache_hit_avoids_recomputing_preprocessing(self, tmp_path, monkeypatch):
        ds = self._build(tmp_path, monkeypatch, [("clipA", "-", "bonafide")])
        # pre-seed the cache so _get_raw_log_mel should hit np.load, not preprocess_clip
        cached_array = np.full((4, 4), 9.0)
        np.save(ds.cache_dir / "clipA.npy", cached_array)

        def fail_if_called(filepath, normalize):
            raise AssertionError("preprocess_clip should not run on a cache hit")

        monkeypatch.setattr(dataset, "preprocess_clip", fail_if_called)
        monkeypatch.setattr(dataset, "normalize_features", lambda log_mel, mean, std: log_mel)

        spec_tensor, _label, _attack_id = ds[0]
        np.testing.assert_array_equal(spec_tensor.squeeze(0).numpy(), cached_array)

    def test_use_cache_false_never_touches_disk_cache(self, tmp_path, monkeypatch):
        ds = self._build(tmp_path, monkeypatch, [("clipA", "-", "bonafide")], use_cache=False)
        calls = []
        monkeypatch.setattr(
            dataset, "preprocess_clip",
            lambda filepath, normalize: calls.append(1) or np.ones((4, 4)),
        )
        monkeypatch.setattr(dataset, "normalize_features", lambda log_mel, mean, std: log_mel)

        ds[0]
        ds[0]  # call twice -- with use_cache=False, both should hit preprocess_clip

        assert len(calls) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
