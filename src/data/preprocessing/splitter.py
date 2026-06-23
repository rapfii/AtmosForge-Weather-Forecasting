"""Chronological data splitting for time series.

CRITICAL: Time series data must NEVER be randomly shuffled.
All splits are strictly chronological: train → val → test.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.data.splitter")


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data chronologically into train/val/test.

    NEVER shuffles — maintains temporal order. The first train_ratio
    fraction goes to training, next val_ratio to validation, and
    the remainder to test.

    Args:
        data: DataFrame with time-ordered rows (DatetimeIndex recommended).
        train_ratio: Fraction for training (default: 0.70).
        val_ratio: Fraction for validation (default: 0.15).

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to ≤ 1.0 or data is empty.

    Example:
        >>> train, val, test = chronological_split(df, 0.70, 0.15)
        >>> # train: 70%, val: 15%, test: 15%
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio < 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) > 1.0"
        )

    if len(data) == 0:
        raise ValueError("Cannot split empty DataFrame")

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    logger.info(
        f"Chronological split: "
        f"train={len(train_df)} ({len(train_df)/n*100:.1f}%), "
        f"val={len(val_df)} ({len(val_df)/n*100:.1f}%), "
        f"test={len(test_df)} ({len(test_df)/n*100:.1f}%)"
    )

    # Verify no temporal leakage
    if hasattr(data, "index") and hasattr(data.index, "max"):
        if isinstance(data.index, pd.DatetimeIndex):
            assert train_df.index.max() < val_df.index.min(), "Temporal leakage: train overlaps val"
            assert val_df.index.max() < test_df.index.min(), "Temporal leakage: val overlaps test"

    return train_df, val_df, test_df
