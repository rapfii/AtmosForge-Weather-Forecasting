"""Sliding window dataset for time series forecasting.

Creates (input, target) pairs from contiguous time series data using
a sliding window approach. Input is a look-back window of all features,
and target is the forecast horizon of the target variable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.data.windower")


class SlidingWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset that creates sliding windows from time series data.

    Generates (input_window, forecast_horizon) pairs by sliding across the
    time series. Each sample consists of:
    - Input: all features for the past `input_window` timesteps
    - Target: target variable for the next `forecast_horizon` timesteps

    Args:
        data: Normalized time series array of shape (n_timesteps, n_features).
        input_window: Number of past timesteps for input (default: 168).
        forecast_horizon: Number of future timesteps to predict (default: 24).
        target_idx: Column index of the target variable in the data array.
        stride: Step size between consecutive windows (default: 1).

    Example:
        >>> dataset = SlidingWindowDataset(data, input_window=168, forecast_horizon=24)
        >>> x, y = dataset[0]
        >>> x.shape  # (168, n_features)
        >>> y.shape  # (24,)
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        input_window: int = 168,
        forecast_horizon: int = 24,
        target_idx: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__()

        if len(data) < input_window + forecast_horizon:
            raise ValueError(
                f"Data length ({len(data)}) is less than "
                f"input_window ({input_window}) + forecast_horizon ({forecast_horizon})"
            )

        self.data = data.astype(np.float32)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.target_idx = target_idx
        self.stride = stride

        # Precompute valid window indices
        self.n_samples = (len(data) - input_window - forecast_horizon) // stride + 1

        logger.info(
            f"SlidingWindowDataset: {self.n_samples} samples "
            f"(window={input_window}, horizon={forecast_horizon}, "
            f"features={data.shape[1]}, stride={stride})"
        )

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single (input, target) pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - input_tensor: shape (input_window, n_features)
                - target_tensor: shape (forecast_horizon,)
        """
        start = idx * self.stride
        end_input = start + self.input_window
        end_target = end_input + self.forecast_horizon

        # Input: all features for the look-back window
        input_data = self.data[start:end_input, :]

        # Target: only the target variable for the forecast horizon
        target_data = self.data[end_input:end_target, self.target_idx]

        input_tensor = torch.from_numpy(input_data)
        target_tensor = torch.from_numpy(target_data)

        return input_tensor, target_tensor

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self.data.shape[1]

    @property
    def feature_shape(self) -> tuple[int, int]:
        """Shape of a single input sample: (input_window, n_features)."""
        return (self.input_window, self.n_features)
