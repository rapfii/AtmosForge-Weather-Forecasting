"""Weather data normalizer wrapping scikit-learn StandardScaler.

CRITICAL: Scaler must be fit ONLY on training data. Validation and test
sets are transformed using the train-fit scaler to prevent data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.data.normalizer")


class WeatherNormalizer:
    """StandardScaler wrapper for weather time series data.

    Fits a per-column StandardScaler on training data and applies the
    same transformation to validation and test data to prevent leakage.

    Attributes:
        scaler: Underlying sklearn StandardScaler.
        is_fitted: Whether the scaler has been fit on data.
        feature_names: Column names from the training data.

    Example:
        >>> normalizer = WeatherNormalizer()
        >>> train_norm = normalizer.fit_transform(train_df)
        >>> val_norm = normalizer.transform(val_df)  # Uses train statistics
        >>> test_norm = normalizer.transform(test_df)  # Uses train statistics
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.is_fitted: bool = False
        self.feature_names: list[str] = []
        self._means: NDArray[np.floating[Any]] | None = None
        self._stds: NDArray[np.floating[Any]] | None = None

    def fit(self, data: pd.DataFrame) -> "WeatherNormalizer":
        """Fit the scaler on training data ONLY.

        Args:
            data: Training DataFrame (must be numeric columns only).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If data contains non-numeric columns.
        """
        # Ensure numeric only
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) != len(data.columns):
            non_numeric = set(data.columns) - set(numeric_cols)
            logger.warning(
                f"Non-numeric columns will be dropped from normalization: {non_numeric}"
            )

        self.feature_names = numeric_cols
        self.scaler.fit(data[numeric_cols].values)
        self.is_fitted = True
        self._means = self.scaler.mean_  # type: ignore[assignment]
        self._stds = self.scaler.scale_  # type: ignore[assignment]

        logger.info(f"Normalizer fit on {len(numeric_cols)} features")
        return self

    def transform(self, data: pd.DataFrame) -> NDArray[np.floating[Any]]:
        """Transform data using the fitted scaler.

        Args:
            data: DataFrame to transform (must have same columns as fit data).

        Returns:
            Normalized numpy array of shape (n_samples, n_features).

        Raises:
            RuntimeError: If scaler has not been fit yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Normalizer has not been fit. Call fit() or fit_transform() first."
            )

        # Use only the columns that were present during fitting
        available_cols = [c for c in self.feature_names if c in data.columns]
        if len(available_cols) != len(self.feature_names):
            missing = set(self.feature_names) - set(available_cols)
            logger.warning(f"Missing columns during transform: {missing}")

        values: NDArray[np.floating[Any]] = self.scaler.transform(
            data[available_cols].values
        )
        return values

    def fit_transform(self, data: pd.DataFrame) -> NDArray[np.floating[Any]]:
        """Fit scaler on data and transform it.

        Convenience method — equivalent to fit() then transform().

        Args:
            data: Training DataFrame.

        Returns:
            Normalized numpy array.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Inverse transform normalized data back to original scale.

        Args:
            data: Normalized numpy array.

        Returns:
            De-normalized numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fit.")
        result: NDArray[np.floating[Any]] = self.scaler.inverse_transform(data)
        return result

    def save(self, path: str | Path) -> None:
        """Save scaler state to disk.

        Args:
            path: File path to save the scaler (pickle format).
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(  # noqa: S301
                {
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "is_fitted": self.is_fitted,
                },
                f,
            )
        logger.info(f"Normalizer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "WeatherNormalizer":
        """Load scaler state from disk.

        Args:
            path: File path to the saved scaler.

        Returns:
            Loaded WeatherNormalizer instance.
        """
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301

        normalizer = cls()
        normalizer.scaler = state["scaler"]
        normalizer.feature_names = state["feature_names"]
        normalizer.is_fitted = state["is_fitted"]
        normalizer._means = normalizer.scaler.mean_  # type: ignore[assignment]
        normalizer._stds = normalizer.scaler.scale_  # type: ignore[assignment]
        return normalizer
