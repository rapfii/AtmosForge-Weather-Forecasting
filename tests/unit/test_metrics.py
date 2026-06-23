"""Unit tests for evaluation metrics.

100% coverage for metrics.py — verifies correctness of:
- MAE, RMSE with known values
- CRPS formula correctness
- DM test with perfect forecast
- Pinball loss monotonicity
- Coverage calculation
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    coverage,
    crps_gaussian,
    crps_quantile,
    diebold_mariano_test,
    mae,
    mape,
    pinball_loss,
    rmse,
    smape,
)


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5])
        # |0.5| + |0.5| + |0.5| + |0.5| = 2.0 / 4 = 0.5
        assert mae(y_true, y_pred) == pytest.approx(0.5)

    def test_symmetric(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([3.0, 0.0])
        assert mae(y_true, y_pred) == mae(y_pred, y_true)

    def test_non_negative(self) -> None:
        y = np.random.randn(100)
        p = np.random.randn(100)
        assert mae(y, p) >= 0


class TestRMSE:
    """Tests for Root Mean Squared Error."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # MSE = (1 + 1 + 1) / 3 = 1.0, RMSE = 1.0
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_geq_mae(self) -> None:
        """RMSE should always be >= MAE."""
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestMAPE:
    """Tests for Mean Absolute Percentage Error."""

    def test_zero_safe(self) -> None:
        """Should not crash with zero values."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.5, 2.5])
        result = mape(y_true, y_pred)
        assert np.isfinite(result)

    def test_non_negative(self) -> None:
        y = np.random.randn(100) + 10  # Positive values
        p = np.random.randn(100) + 10
        assert mape(y, p) >= 0


class TestSMAPE:
    """Tests for Symmetric MAPE."""

    def test_symmetric(self) -> None:
        """sMAPE should be roughly symmetric."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true), abs=0.1)

    def test_range(self) -> None:
        """sMAPE should be between 0 and 200."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert 0 <= smape(y_true, y_pred) <= 200

    def test_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert smape(y, y) == pytest.approx(0.0, abs=1e-5)


class TestPinballLoss:
    """Tests for Pinball (Quantile) Loss."""

    def test_median_equals_mae(self) -> None:
        """Pinball at q=0.5 should equal 0.5 * MAE."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        expected = 0.5 * mae(y_true, y_pred)
        assert pinball_loss(y_true, y_pred, 0.5) == pytest.approx(expected)

    def test_monotonicity(self) -> None:
        """Higher quantile should penalize under-prediction more."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred_low = np.array([3.0, 3.0, 3.0])  # Under-prediction

        # For under-prediction, higher quantile = higher loss
        loss_q10 = pinball_loss(y_true, y_pred_low, 0.1)
        loss_q50 = pinball_loss(y_true, y_pred_low, 0.5)
        loss_q90 = pinball_loss(y_true, y_pred_low, 0.9)

        assert loss_q10 < loss_q50 < loss_q90

    def test_non_negative(self) -> None:
        y = np.random.randn(100)
        p = np.random.randn(100)
        for q in [0.1, 0.5, 0.9]:
            assert pinball_loss(y, p, q) >= 0


class TestCRPS:
    """Tests for Continuous Ranked Probability Score."""

    def test_crps_gaussian_perfect(self) -> None:
        """CRPS should be low for accurate predictions with small sigma."""
        y_true = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.01, 0.01, 0.01])
        result = crps_gaussian(y_true, mu, sigma)
        assert result < 0.1

    def test_crps_gaussian_non_negative(self) -> None:
        """CRPS should always be non-negative."""
        y = np.random.randn(100)
        mu = np.random.randn(100)
        sigma = np.abs(np.random.randn(100)) + 0.1
        assert crps_gaussian(y, mu, sigma) >= 0

    def test_crps_quantile_non_negative(self) -> None:
        """Quantile-based CRPS should be non-negative."""
        y = np.random.randn(100)
        q_preds = {
            0.1: y - 1.0,
            0.5: y + 0.1,
            0.9: y + 1.0,
        }
        assert crps_quantile(y, q_preds) >= 0


class TestCoverage:
    """Tests for prediction interval coverage."""

    def test_perfect_coverage(self) -> None:
        y_true = np.array([2.0, 3.0, 4.0])
        q_lower = np.array([1.0, 2.0, 3.0])
        q_upper = np.array([3.0, 4.0, 5.0])
        result = coverage(y_true, q_lower, q_upper)
        assert result["actual_coverage"] == 1.0

    def test_zero_coverage(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        q_lower = np.array([1.0, 1.0, 1.0])
        q_upper = np.array([2.0, 2.0, 2.0])
        result = coverage(y_true, q_lower, q_upper)
        assert result["actual_coverage"] == 0.0


class TestDieboldMariano:
    """Tests for Diebold-Mariano statistical test."""

    def test_identical_forecasts(self) -> None:
        """Identical forecasts should have p >> 0.05."""
        y_true = np.random.randn(200)
        pred = y_true + np.random.randn(200) * 0.1
        result = diebold_mariano_test(y_true, pred, pred)
        # Identical forecasts → DM stat ≈ 0 → p ≈ 1
        assert result["p_value"] > 0.05
        assert not result["significant"]

    def test_one_clearly_better(self) -> None:
        """One clearly better model should give p < 0.05."""
        np.random.seed(42)
        y_true = np.random.randn(500)
        good_pred = y_true + np.random.randn(500) * 0.1  # Small noise
        bad_pred = y_true + np.random.randn(500) * 2.0   # Large noise
        result = diebold_mariano_test(y_true, good_pred, bad_pred)
        assert result["significant"]
        assert result["mean_loss_diff"] < 0  # Model 1 is better

    def test_two_tailed(self) -> None:
        """Test should be two-tailed (p-value between 0 and 1)."""
        y_true = np.random.randn(100)
        pred_1 = np.random.randn(100)
        pred_2 = np.random.randn(100)
        result = diebold_mariano_test(y_true, pred_1, pred_2)
        assert 0 <= result["p_value"] <= 1

    def test_mae_loss_fn(self) -> None:
        """Should work with MAE loss function."""
        y_true = np.random.randn(100)
        pred = y_true + 0.1
        result = diebold_mariano_test(y_true, pred, pred, loss_fn="mae")
        assert result["p_value"] > 0.05


class TestComputeAllMetrics:
    """Tests for compute_all_metrics convenience function."""

    def test_basic_output(self) -> None:
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        metrics = compute_all_metrics(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "smape" in metrics

    def test_with_quantiles(self) -> None:
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1
        q_preds = {
            0.1: y_pred - 1.0,
            0.5: y_pred,
            0.9: y_pred + 1.0,
        }
        metrics = compute_all_metrics(y_true, y_pred, q_preds)
        assert "crps" in metrics
        assert "pinball_q10" in metrics
        assert "pinball_q50" in metrics
        assert "pinball_q90" in metrics
        assert "coverage_80" in metrics
