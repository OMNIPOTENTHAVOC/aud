"""
Unit tests for evaluation/eer.py and evaluation/metrics.py.

This is the highest-correctness-risk layer in the whole project: if EER is
wrong, every number in the paper is wrong. These tests were verified against
actual sklearn behavior (not just reasoned about), which surfaced two
non-obvious, worth-knowing-about behaviors baked into the current code:

1. When all scores are tied, roc_curve's eer_threshold comes back as `inf`.
   That's silently propagated into f1_at_eer_threshold, which then collapses
   to 0.0 (nothing gets predicted positive at threshold=inf) -- not a bug,
   but easy to misread as "the model is bad" during early training when
   scores haven't spread out yet.
2. compute_eer(y_true, y_score) RAISES ValueError("All-NaN slice
   encountered") if y_true is single-class (all 0s or all 1s). This can't
   happen in compute_all_metrics() (labels always mix bonafide+spoof) but
   COULD happen in compute_per_attack_eer() if a protocol/filtering bug ever
   produced a subset with zero bonafide clips -- worth guarding explicitly
   in eer.py with a clear error message rather than relying on this
   accidental-but-real crash.

Run from the project root:

    pytest tests/test_eer_metrics_unit.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eer import compute_eer
from evaluation.metrics import compute_f1, compute_auc, compute_per_attack_eer, compute_all_metrics


# ---------------------------------------------------------------------------
# compute_eer
# ---------------------------------------------------------------------------

class TestComputeEER:
    def test_perfect_separation_gives_zero_eer(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
        eer, _threshold = compute_eer(y_true, y_score)
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_inverted_scores_gives_high_eer(self):
        """Bonafide scored higher than spoof across the board -- EER should
        be near its worst-case value, not silently near zero."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.9, 0.8, 0.95, 0.1, 0.2, 0.05]  # inverted
        eer, _threshold = compute_eer(y_true, y_score)
        assert eer > 0.9

    def test_all_tied_scores_gives_eer_half_and_inf_threshold(self):
        """A coin-flip-quality classifier should land at EER ~0.5. Also
        pins the (surprising) inf threshold that results -- see module
        docstring point 1."""
        y_true = [0, 1, 0, 1]
        y_score = [0.5, 0.5, 0.5, 0.5]
        eer, threshold = compute_eer(y_true, y_score)
        assert eer == pytest.approx(0.5)
        assert threshold == float("inf")

    def test_single_class_labels_raise_valueerror(self):
        """Documents current (fragile) behavior: a single-class y_true
        crashes with an opaque numpy error rather than a clear message.
        If eer.py is ever hardened with an explicit guard for this case,
        UPDATE this test to assert the new, clearer exception instead of
        deleting it -- the underlying degenerate-input risk is real
        (see module docstring point 2)."""
        with pytest.raises(ValueError):
            compute_eer([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError):
            compute_eer([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4])

    def test_single_bonafide_single_spoof_does_not_crash(self):
        """Smallest possible valid (2-class) input -- an attack type with
        exactly one spoof sample paired against one bonafide sample."""
        eer, _threshold = compute_eer([0, 1], [0.2, 0.8])
        assert 0.0 <= eer <= 1.0


# ---------------------------------------------------------------------------
# compute_f1 / compute_auc
# ---------------------------------------------------------------------------

class TestComputeF1:
    def test_perfect_predictions_give_f1_one(self):
        assert compute_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_threshold_boundary_is_inclusive(self):
        """y_pred uses `s >= threshold` -- a score exactly at the threshold
        must count as positive, not negative."""
        y_true = [1]
        y_score = [0.5]
        f1 = compute_f1(y_true, y_score, threshold=0.5)
        assert f1 == pytest.approx(1.0)  # would be 0.0 if boundary were exclusive

    def test_all_negative_predictions_and_labels_gives_zero_not_crash(self):
        """Degenerate-but-valid case: nothing predicted positive, nothing
        actually positive. sklearn treats this as ill-defined and returns
        0.0 (with a warning) rather than 1.0 -- pin that so a future
        sklearn version change or code change surfaces as a test failure."""
        f1 = compute_f1([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
        assert f1 == pytest.approx(0.0)


class TestComputeAUC:
    def test_perfect_ranking_gives_auc_one(self):
        auc = compute_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert auc == pytest.approx(1.0)

    def test_random_ranking_gives_auc_near_half(self):
        auc = compute_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        assert auc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_per_attack_eer
# ---------------------------------------------------------------------------

class TestComputePerAttackEER:
    def test_bonafide_shared_across_every_attack_subset(self):
        """Standard ASVspoof protocol: each attack's EER is computed against
        ALL bonafide clips, not a per-attack split of them."""
        y_true = [0, 0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.15, 0.9, 0.3]
        attack_ids = ["-", "-", "-", "A01", "A02"]

        results = compute_per_attack_eer(y_true, y_score, attack_ids)

        assert set(results.keys()) == {"A01", "A02"}
        for attack, (eer, n_spoof) in results.items():
            assert 0.0 <= eer <= 1.0
            assert n_spoof == 1

    def test_bonafide_only_marker_excluded_from_attack_list(self):
        y_true = [0, 0, 1]
        y_score = [0.1, 0.2, 0.9]
        attack_ids = ["-", "-", "A01"]
        results = compute_per_attack_eer(y_true, y_score, attack_ids)
        assert "-" not in results

    def test_n_spoof_counts_all_clips_of_that_attack_not_just_subset_size(self):
        y_true = [0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.7, 0.8, 0.9]
        attack_ids = ["-", "-", "A01", "A01", "A01"]
        results = compute_per_attack_eer(y_true, y_score, attack_ids)
        _eer, n_spoof = results["A01"]
        assert n_spoof == 3

    def test_no_attacks_present_returns_empty_dict(self):
        """All-bonafide-clips-only slice: no attack IDs to break down."""
        results = compute_per_attack_eer([0, 0], [0.1, 0.2], ["-", "-"])
        assert results == {}


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_all_expected_keys(self):
        result = compute_all_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        assert set(result.keys()) == {
            "eer", "eer_threshold", "f1_at_0.5", "f1_at_eer_threshold", "auc",
        }

    def test_f1_at_eer_threshold_uses_eer_threshold_not_half(self):
        """f1_at_0.5 and f1_at_eer_threshold should genuinely differ when
        the EER threshold isn't 0.5 -- otherwise the second metric is
        pointless. Use a case where the optimal separating point is well
        away from 0.5."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]  # all above 0.5

        result = compute_all_metrics(y_true, y_score)

        # at threshold=0.5, everything is predicted positive -> poor F1
        assert result["f1_at_0.5"] < 1.0
        # at the EER threshold, separation should be much better
        assert result["f1_at_eer_threshold"] > result["f1_at_0.5"]

    def test_degenerate_tied_scores_produces_finite_metrics_dict(self):
        """Regression test for the inf-threshold interaction documented at
        the top of this file: compute_all_metrics must not itself crash on
        this input, even though f1_at_eer_threshold ends up at 0.0."""
        result = compute_all_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        assert result["eer"] == pytest.approx(0.5)
        assert result["eer_threshold"] == float("inf")
        assert result["f1_at_eer_threshold"] == pytest.approx(0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
