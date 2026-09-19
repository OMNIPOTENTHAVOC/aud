"""
Unit tests for datasets_src/preproc.py.

These mock librosa's I/O and DSP calls (no real .flac files or network
needed) so they run fast and deterministically anywhere. They complement,
not replace, tests/test_preprocessing.py's existing end-to-end smoke test
against a real clip.

Run from the project root:

    pytest tests/test_preproc_unit.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config
from datasets_src import preproc


# ---------------------------------------------------------------------------
# load_waveform
# ---------------------------------------------------------------------------

class TestLoadWaveform:
    def test_loads_at_configured_sample_rate(self, monkeypatch):
        captured = {}

        def fake_load(filepath, sr):
            captured["filepath"] = filepath
            captured["sr"] = sr
            return np.zeros(1000, dtype=np.float64), sr

        monkeypatch.setattr(preproc.librosa, "load", fake_load)

        waveform = preproc.load_waveform("dummy.flac")

        assert captured["sr"] == config.SAMPLE_RATE
        assert waveform.dtype == np.float32

    def test_returns_float32_regardless_of_source_dtype(self, monkeypatch):
        monkeypatch.setattr(
            preproc.librosa, "load",
            lambda filepath, sr: (np.ones(10, dtype=np.float64) * 0.5, sr),
        )
        waveform = preproc.load_waveform("dummy.flac")
        assert waveform.dtype == np.float32
        np.testing.assert_allclose(waveform, 0.5)


# ---------------------------------------------------------------------------
# trim_silence
# ---------------------------------------------------------------------------

class TestTrimSilence:
    def test_passes_top_db_through(self, monkeypatch):
        captured = {}

        def fake_trim(waveform, top_db):
            captured["top_db"] = top_db
            return waveform, (0, len(waveform))

        monkeypatch.setattr(preproc.librosa.effects, "trim", fake_trim)
        preproc.trim_silence(np.zeros(100), top_db=30)
        assert captured["top_db"] == 30

    def test_default_top_db_is_25(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            preproc.librosa.effects, "trim",
            lambda waveform, top_db: captured.setdefault("top_db", top_db) or (waveform, None),
        )
        preproc.trim_silence(np.zeros(100))
        assert captured["top_db"] == 25


# ---------------------------------------------------------------------------
# normalize_amplitude
# ---------------------------------------------------------------------------

class TestNormalizeAmplitude:
    def test_scales_peak_to_one(self):
        waveform = np.array([0.0, -2.0, 4.0, 1.0], dtype=np.float32)
        result = preproc.normalize_amplitude(waveform)
        assert np.max(np.abs(result)) == pytest.approx(1.0)
        np.testing.assert_allclose(result, waveform / 4.0)

    def test_all_zero_waveform_does_not_divide_by_zero(self):
        """Edge case: a fully silent clip has peak == 0. Must return the
        waveform unchanged rather than raising or producing NaN/inf."""
        waveform = np.zeros(50, dtype=np.float32)
        result = preproc.normalize_amplitude(waveform)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_array_equal(result, waveform)

    def test_negative_peak_handled_via_abs(self):
        waveform = np.array([-5.0, 1.0, 2.0], dtype=np.float32)
        result = preproc.normalize_amplitude(waveform)
        assert np.max(np.abs(result)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pad_or_truncate
# ---------------------------------------------------------------------------

class TestPadOrTruncate:
    def test_pads_short_waveform_with_zeros(self):
        waveform = np.ones(10, dtype=np.float32)
        result = preproc.pad_or_truncate(waveform, target_len=20)
        assert len(result) == 20
        np.testing.assert_array_equal(result[:10], np.ones(10))
        np.testing.assert_array_equal(result[10:], np.zeros(10))

    def test_truncates_long_waveform(self):
        waveform = np.arange(100, dtype=np.float32)
        result = preproc.pad_or_truncate(waveform, target_len=30)
        assert len(result) == 30
        np.testing.assert_array_equal(result, waveform[:30])

    def test_exact_length_passes_through_unchanged(self):
        waveform = np.arange(50, dtype=np.float32)
        result = preproc.pad_or_truncate(waveform, target_len=50)
        np.testing.assert_array_equal(result, waveform)

    def test_empty_waveform_pads_to_full_length(self):
        """Edge case: a zero-length waveform (e.g. trim_silence stripped
        everything) must still produce a fixed-length, zero-padded array
        rather than crashing downstream in melspectrogram extraction."""
        waveform = np.array([], dtype=np.float32)
        result = preproc.pad_or_truncate(waveform, target_len=16)
        assert len(result) == 16
        np.testing.assert_array_equal(result, np.zeros(16))

    def test_falls_back_to_config_max_audio_length(self):
        waveform = np.ones(5, dtype=np.float32)
        result = preproc.pad_or_truncate(waveform, target_len=None)
        assert len(result) == config.MAX_AUDIO_LENGTH


# ---------------------------------------------------------------------------
# extract_logmel_spectrogram
# ---------------------------------------------------------------------------

class TestExtractLogmelSpectrogram:
    def test_passes_config_params_to_melspectrogram(self, monkeypatch):
        captured = {}

        def fake_melspectrogram(y, sr, n_fft, win_length, hop_length, n_mels, power):
            captured.update(sr=sr, n_fft=n_fft, win_length=win_length,
                             hop_length=hop_length, n_mels=n_mels, power=power)
            return np.ones((n_mels, 10))

        monkeypatch.setattr(preproc.librosa.feature, "melspectrogram", fake_melspectrogram)
        monkeypatch.setattr(preproc.librosa, "power_to_db", lambda mel, ref: mel)

        preproc.extract_logmel_spectrogram(np.zeros(1000))

        assert captured["sr"] == config.SAMPLE_RATE
        assert captured["n_fft"] == config.N_FFT
        assert captured["win_length"] == config.WIN_LENGTH
        assert captured["hop_length"] == config.HOP_LENGTH
        assert captured["n_mels"] == config.N_MELS
        assert captured["power"] == 2.0

    def test_output_shape_is_n_mels_by_time(self, monkeypatch):
        monkeypatch.setattr(
            preproc.librosa.feature, "melspectrogram",
            lambda **kwargs: np.ones((config.N_MELS, 37)),
        )
        monkeypatch.setattr(preproc.librosa, "power_to_db", lambda mel, ref: mel)

        out = preproc.extract_logmel_spectrogram(np.zeros(1000))
        assert out.shape == (config.N_MELS, 37)


# ---------------------------------------------------------------------------
# normalize_features
# ---------------------------------------------------------------------------

class TestNormalizeFeatures:
    def test_uses_provided_mean_and_std(self):
        log_mel = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = preproc.normalize_features(log_mel, mean=10.0, std=10.0)
        expected = (log_mel - 10.0) / (10.0 + 1e-8)
        np.testing.assert_allclose(result, expected)

    def test_falls_back_to_per_sample_stats_when_none(self):
        """Documented as an ad-hoc-only fallback -- verify it actually
        zero-means/unit-variances the input when mean/std are omitted."""
        log_mel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = preproc.normalize_features(log_mel, mean=None, std=None)
        assert result.mean() == pytest.approx(0.0, abs=1e-6)
        assert result.std() == pytest.approx(1.0, abs=1e-6)

    def test_zero_std_does_not_divide_by_zero(self):
        """Edge case: a perfectly flat spectrogram (std == 0) must not
        produce inf/NaN, thanks to the +1e-8 epsilon."""
        log_mel = np.full((4, 4), 7.0)
        result = preproc.normalize_features(log_mel, mean=7.0, std=0.0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-3)

    def test_mean_only_provided_still_triggers_fallback(self):
        """mean/std are only used if BOTH are non-None (`if mean is None or
        std is None`) -- passing just one silently falls back to per-sample
        stats for both. This is easy to hit by accident; pin the behavior."""
        log_mel = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = preproc.normalize_features(log_mel, mean=999.0, std=None)
        # If it had used mean=999 with a per-sample std, values would be
        # wildly negative. Confirm it instead used the true per-sample mean.
        expected = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# preprocess_clip (full pipeline, all stages mocked)
# ---------------------------------------------------------------------------

class TestPreprocessClip:
    @pytest.fixture
    def mock_pipeline(self, monkeypatch):
        monkeypatch.setattr(preproc, "load_waveform", lambda filepath: np.ones(2000, dtype=np.float32))
        monkeypatch.setattr(preproc, "trim_silence", lambda waveform, top_db=25: waveform)
        monkeypatch.setattr(preproc, "normalize_amplitude", lambda waveform: waveform)
        monkeypatch.setattr(preproc, "pad_or_truncate", lambda waveform, target_len=None: waveform)
        monkeypatch.setattr(
            preproc, "extract_logmel_spectrogram",
            lambda waveform: np.full((config.N_MELS, 5), 3.0),
        )

    def test_normalize_true_applies_feature_normalization(self, mock_pipeline):
        result = preproc.preprocess_clip("dummy.flac", mean=3.0, std=1.0, normalize=True)
        np.testing.assert_allclose(result, np.zeros((config.N_MELS, 5)), atol=1e-3)

    def test_normalize_false_returns_raw_logmel(self, mock_pipeline):
        """This is the path compute_train_stats() relies on -- must return
        genuinely unnormalized values, not accidentally pre-normalized ones."""
        result = preproc.preprocess_clip("dummy.flac", normalize=False)
        np.testing.assert_array_equal(result, np.full((config.N_MELS, 5), 3.0))

    def test_stages_called_in_order(self, monkeypatch):
        order = []
        monkeypatch.setattr(preproc, "load_waveform", lambda fp: order.append("load") or np.ones(10))
        monkeypatch.setattr(preproc, "trim_silence", lambda w: order.append("trim") or w)
        monkeypatch.setattr(preproc, "normalize_amplitude", lambda w: order.append("norm_amp") or w)
        monkeypatch.setattr(preproc, "pad_or_truncate", lambda w: order.append("pad") or w)
        monkeypatch.setattr(preproc, "extract_logmel_spectrogram", lambda w: order.append("logmel") or np.ones((4, 4)))

        preproc.preprocess_clip("dummy.flac", normalize=False)

        assert order == ["load", "trim", "norm_amp", "pad", "logmel"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
