"""Base weather dataset abstract class for all AtmosForge data sources.

All dataset implementations must inherit from BaseWeatherDataset and implement
the required abstract methods for downloading, preprocessing, and splitting.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader


class BaseWeatherDataset(abc.ABC):
    """Abstract base class for weather dataset implementations.

    Provides a standardized interface for downloading, preprocessing, and
    creating PyTorch DataLoaders from meteorological datasets. All concrete
    implementations (Jena, Open-Meteo, ERA5) must inherit from this class.

    The pipeline flow:
        download() → preprocess() → get_splits() → DataLoaders

    Args:
        data_dir: Root directory for storing raw and processed data.
        target_column: Name of the target variable to forecast.
        input_window: Number of past timesteps used as input.
        forecast_horizon: Number of future timesteps to predict.
        train_ratio: Fraction of data for training (default: 0.70).
        val_ratio: Fraction of data for validation (default: 0.15).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader worker processes.

    Example:
        >>> class JenaDataset(BaseWeatherDataset):
        ...     def download(self) -> Path:
        ...         # Download and cache Jena Climate CSV
        ...         ...
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        target_column: str = "T (degC)",
        input_window: int = 168,
        forecast_horizon: int = 24,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.target_column = target_column
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Validate ratios
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")
        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}")
        if self.test_ratio <= 0:
            raise ValueError(
                f"test_ratio must be > 0, got {self.test_ratio:.2f} "
                f"(train={self.train_ratio}, val={self.val_ratio})"
            )

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def download(self) -> Path:
        """Download raw data and cache locally.

        Should check if data already exists before re-downloading.
        Raw data is stored in self.raw_dir.

        Returns:
            Path to the downloaded raw data file.
        """
        ...

    @abc.abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Preprocess raw data into a clean DataFrame.

        Steps typically include:
        - Parse timestamps
        - Handle missing values (forward-fill + boolean flag)
        - Select relevant columns
        - Resample if needed

        Returns:
            Preprocessed pandas DataFrame with DatetimeIndex.
        """
        ...

    @abc.abstractmethod
    def get_splits(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create chronological train/val/test DataLoaders.

        CRITICAL: Splits must be chronological — NEVER random shuffle.
        Normalization (StandardScaler) must be fit ONLY on train split.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Return dataset configuration as a dictionary.

        Useful for MLflow parameter logging.

        Returns:
            Dictionary of dataset configuration parameters.
        """
        return {
            "dataset_class": self.__class__.__name__,
            "data_dir": str(self.data_dir),
            "target_column": self.target_column,
            "input_window": self.input_window,
            "forecast_horizon": self.forecast_horizon,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"{self.__class__.__name__}("
            f"target='{self.target_column}', "
            f"window={self.input_window}, "
            f"horizon={self.forecast_horizon}, "
            f"batch_size={self.batch_size})"
        )
