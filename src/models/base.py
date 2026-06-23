"""Base forecaster abstract class for all AtmosForge models.

All model implementations must inherit from BaseForecaster and implement
the required abstract methods. This ensures a consistent interface across
baseline and advanced architectures.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class ForecastOutput:
    """Standardized output container for all forecasting models.

    Attributes:
        point_forecast: Point predictions of shape (batch_size, horizon).
        quantiles: Dict mapping quantile levels to predictions.
            Expected keys: 0.1 (q10), 0.5 (q50), 0.9 (q90).
            Each value has shape (batch_size, horizon).
        metadata: Optional dictionary for model-specific metadata
            (e.g., attention weights, intermediate representations).
    """

    point_forecast: torch.Tensor
    quantiles: dict[float, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate output shapes are consistent."""
        if self.quantiles:
            for q_level, q_pred in self.quantiles.items():
                if q_pred.shape != self.point_forecast.shape:
                    raise ValueError(
                        f"Quantile {q_level} shape {q_pred.shape} does not match "
                        f"point forecast shape {self.point_forecast.shape}"
                    )


class BaseForecaster(nn.Module, abc.ABC):
    """Abstract base class for all AtmosForge forecasting models.

    Every model in the framework — whether a baseline (LSTM, GRU, CNN-1D) or
    an advanced architecture (TFT, N-HiTS, PatchTST) — must inherit from this
    class and implement the required abstract methods.

    This ensures:
    - Consistent input/output interface across all models
    - Unified training loop via GenericTrainer
    - Standardized quantile output for uncertainty quantification

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Base hidden dimension size.
        output_size: Number of output features (typically 1 for temperature).
        horizon: Forecast horizon (number of future steps to predict).
        quantiles: List of quantile levels for probabilistic output.
            Default: [0.1, 0.5, 0.9] for 80% prediction interval.
        dropout: Dropout rate for regularization.

    Example:
        >>> class MyModel(BaseForecaster):
        ...     def forward(self, x):
        ...         # x shape: (batch, seq_len, n_features)
        ...         return ForecastOutput(point_forecast=predictions)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        horizon: int = 24,
        quantiles: list[float] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon = horizon
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.dropout = dropout

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> ForecastOutput:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features).

        Returns:
            ForecastOutput containing point forecast and optional quantile
            predictions, each of shape (batch_size, horizon).
        """
        ...

    @abc.abstractmethod
    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and optional LR scheduler.

        Returns:
            Dictionary with keys:
                - 'optimizer': torch.optim.Optimizer instance
                - 'scheduler' (optional): LR scheduler instance
                - 'monitor' (optional): metric to monitor for scheduler

        Example:
            >>> return {
            ...     'optimizer': torch.optim.Adam(self.parameters(), lr=1e-3),
            ...     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            ...         optimizer, patience=5
            ...     ),
            ...     'monitor': 'val_loss',
            ... }
        """
        ...

    @abc.abstractmethod
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute training loss for a single batch.

        Args:
            batch: Tuple of (input, target) tensors where:
                - input shape: (batch_size, seq_len, n_features)
                - target shape: (batch_size, horizon)

        Returns:
            Scalar loss tensor.
        """
        ...

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute validation metrics for a single batch.

        Default implementation computes MSE loss. Override for custom
        validation metrics.

        Args:
            batch: Tuple of (input, target) tensors.

        Returns:
            Dictionary mapping metric names to scalar tensors.
        """
        x, y = batch
        output = self.forward(x)
        val_loss = nn.functional.mse_loss(output.point_forecast, y)
        return {"val_loss": val_loss}

    def predict(self, x: torch.Tensor) -> ForecastOutput:
        """Generate predictions with the model in eval mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features).

        Returns:
            ForecastOutput with point forecast and quantile estimates.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return string representation with parameter count."""
        params = self.count_parameters()
        param_str = f"{params / 1e6:.2f}M" if params >= 1e6 else f"{params / 1e3:.1f}K"
        return (
            f"{self.__class__.__name__}("
            f"input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"horizon={self.horizon}, "
            f"params={param_str})"
        )
