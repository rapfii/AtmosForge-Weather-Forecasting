"""Evaluation metrics for time series forecasting.

Implements deterministic metrics (MAE, RMSE, MAPE, sMAPE),
probabilistic metrics (CRPS, Pinball Loss, Coverage), and
statistical comparison (Diebold-Mariano test).

References:
- CRPS: Gneiting & Raftery (2007). Strictly Proper Scoring Rules.
- DM Test: Diebold & Mariano (1995). Comparing Predictive Accuracy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.evaluation.metrics")


# ═══════════════════════════════════════════════════════════════
# DETERMINISTIC METRICS
# ═══════════════════════════════════════════════════════════════


def mae(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE score (lower is better).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE score (lower is better).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error.

    Handles zero-division by adding epsilon to denominator.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        MAPE score as percentage (lower is better).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def smape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    epsilon: float = 1e-8,
) -> float:
    """Symmetric Mean Absolute Percentage Error.

    More stable than MAPE when values are near zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        sMAPE score as percentage (lower is better, range: 0-200).
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return float(np.mean(numerator / denominator) * 100)


# ═══════════════════════════════════════════════════════════════
# PROBABILISTIC METRICS
# ═══════════════════════════════════════════════════════════════


def pinball_loss(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    quantile: float,
) -> float:
    """Pinball (Quantile) Loss.

    Asymmetric loss that penalizes over/under-prediction differently
    based on the target quantile.

    Args:
        y_true: Ground truth values.
        y_pred: Quantile prediction values.
        quantile: Target quantile level (e.g., 0.1, 0.5, 0.9).

    Returns:
        Pinball loss (lower is better).
    """
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))


def crps_gaussian(
    y_true: NDArray[np.floating],
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> float:
    """Continuous Ranked Probability Score for Gaussian predictions.

    Proper scoring rule from Gneiting & Raftery (2007).
    CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ, Φ is CDF, φ is PDF.

    Args:
        y_true: Ground truth values.
        mu: Predicted mean (point forecast).
        sigma: Predicted standard deviation.

    Returns:
        Average CRPS (lower is better).
    """
    sigma = np.maximum(sigma, 1e-8)  # Avoid division by zero
    z = (y_true - mu) / sigma

    crps_values = sigma * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )

    return float(np.mean(crps_values))


def crps_quantile(
    y_true: NDArray[np.floating],
    quantile_predictions: dict[float, NDArray[np.floating]],
) -> float:
    """CRPS approximation from quantile predictions.

    Approximates CRPS using quantile predictions (q10, q50, q90).
    Based on the quantile score decomposition.

    Args:
        y_true: Ground truth values.
        quantile_predictions: Dict mapping quantile levels to predictions.

    Returns:
        Approximate CRPS (lower is better).
    """
    total_loss = 0.0
    n_quantiles = len(quantile_predictions)

    for q_level, q_pred in sorted(quantile_predictions.items()):
        total_loss += pinball_loss(y_true, q_pred, q_level)

    return total_loss / max(n_quantiles, 1)


def coverage(
    y_true: NDArray[np.floating],
    q_lower: NDArray[np.floating],
    q_upper: NDArray[np.floating],
    expected_coverage: float = 0.80,
) -> dict[str, float]:
    """Prediction interval coverage.

    Checks what fraction of true values fall within [q_lower, q_upper].
    For q10/q90 interval, expected coverage is 80%.

    Args:
        y_true: Ground truth values.
        q_lower: Lower bound predictions (e.g., q10).
        q_upper: Upper bound predictions (e.g., q90).
        expected_coverage: Expected coverage rate (default: 0.80).

    Returns:
        Dict with 'actual_coverage', 'expected_coverage', and
        'coverage_error' (actual - expected).
    """
    inside = np.logical_and(y_true >= q_lower, y_true <= q_upper)
    actual = float(np.mean(inside))

    return {
        "actual_coverage": actual,
        "expected_coverage": expected_coverage,
        "coverage_error": actual - expected_coverage,
    }


# ═══════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════


def diebold_mariano_test(
    y_true: NDArray[np.floating],
    pred_1: NDArray[np.floating],
    pred_2: NDArray[np.floating],
    horizon: int = 1,
    loss_fn: str = "mse",
) -> dict[str, float]:
    """Diebold-Mariano test for comparing predictive accuracy.

    Two-tailed test. H0: Both forecasts have equal predictive accuracy.
    If p < 0.05, the forecasts are significantly different.

    Reference:
        Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive
        Accuracy. Journal of Business & Economic Statistics.

    Args:
        y_true: Ground truth values.
        pred_1: Predictions from model 1.
        pred_2: Predictions from model 2.
        horizon: Forecast horizon for Newey-West bandwidth.
        loss_fn: Loss function ('mse' or 'mae').

    Returns:
        Dict with 'dm_statistic', 'p_value', 'significant' (p < 0.05).
    """
    n = len(y_true)

    # Compute loss differentials
    if loss_fn == "mse":
        loss_1 = (y_true - pred_1) ** 2
        loss_2 = (y_true - pred_2) ** 2
    elif loss_fn == "mae":
        loss_1 = np.abs(y_true - pred_1)
        loss_2 = np.abs(y_true - pred_2)
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}. Use 'mse' or 'mae'.")

    d = loss_1 - loss_2  # Loss differential

    # Mean and variance of loss differentials
    d_mean = np.mean(d)

    # Newey-West variance estimator
    # Autocovariance at lag k
    gamma_0 = np.mean((d - d_mean) ** 2)

    # Bandwidth = horizon - 1 (as per DM original paper)
    bandwidth = max(horizon - 1, 0)

    nw_var = gamma_0
    for k in range(1, bandwidth + 1):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        nw_var += 2 * (1 - k / (bandwidth + 1)) * gamma_k

    # DM statistic
    dm_stat = d_mean / np.sqrt(max(nw_var / n, 1e-12))

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "mean_loss_diff": float(d_mean),
    }


def compute_all_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    quantile_predictions: dict[float, NDArray[np.floating]] | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics at once.

    Args:
        y_true: Ground truth values.
        y_pred: Point predictions.
        quantile_predictions: Optional dict of quantile predictions.

    Returns:
        Dictionary of all computed metrics.
    """
    metrics: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }

    if quantile_predictions:
        # CRPS from quantiles
        metrics["crps"] = crps_quantile(y_true, quantile_predictions)

        # Pinball loss per quantile
        for q_level, q_pred in quantile_predictions.items():
            metrics[f"pinball_q{int(q_level*100)}"] = pinball_loss(
                y_true, q_pred, q_level
            )

        # Coverage (q10 to q90 = 80% interval)
        if 0.1 in quantile_predictions and 0.9 in quantile_predictions:
            cov = coverage(
                y_true,
                quantile_predictions[0.1],
                quantile_predictions[0.9],
            )
            metrics["coverage_80"] = cov["actual_coverage"]
            metrics["coverage_error"] = cov["coverage_error"]

    return metrics
