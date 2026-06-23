"""LSTM forecaster with residual connections and variational dropout.

Reference:
    Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
    Neural Computation, 9(8), 1735-1780.
    https://doi.org/10.1162/neco.1997.9.8.1735

Architecture:
    Multi-layer LSTM with residual connections between layers and
    variational dropout (same mask across timesteps). Output from the
    last hidden state is projected through point and quantile heads.

Approximate parameters:
    ~0.5M parameters with default config (2 layers, 128 hidden)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster, ForecastOutput


class VariationalDropout(nn.Module):
    """Variational dropout — same mask applied across timesteps.

    Unlike standard dropout which samples a new mask per timestep,
    variational dropout uses the same mask for the entire sequence,
    which is more appropriate for recurrent networks.

    Args:
        dropout: Dropout probability.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply variational dropout.

        Args:
            x: Input of shape (batch, seq_len, features).

        Returns:
            Dropped-out tensor of same shape.
        """
        if not self.training or self.dropout == 0:
            return x

        # Create mask for (batch, 1, features) — broadcast across seq_len
        mask = torch.bernoulli(
            torch.ones(x.size(0), 1, x.size(2), device=x.device) * (1 - self.dropout)
        )
        mask = mask / (1 - self.dropout)  # Scale to maintain expected values
        return x * mask


class LSTMForecaster(BaseForecaster):
    """LSTM with residual connections for time series forecasting.

    Reference:
        Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
        Neural Computation, 9(8), 1735-1780.

    Architecture:
        Multi-layer LSTM with:
        - Residual connections between layers (when hidden_size matches)
        - Variational dropout (consistent mask across timesteps)
        - Layer normalization on hidden states
        - Point forecast head + separate quantile heads

    Args:
        input_size: Number of input features per timestep.
        hidden_size: LSTM hidden state dimension (default: 128).
        output_size: Number of output features (default: 1).
        horizon: Forecast horizon (default: 24).
        num_layers: Number of LSTM layers (default: 2).
        dropout: Dropout rate (default: 0.1).
        bidirectional: Use bidirectional LSTM (default: False).
        quantiles: Quantile levels for probabilistic output.

    Approximate parameters:
        ~0.5M parameters with default config
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        output_size: int = 1,
        horizon: int = 24,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        quantiles: list[float] | None = None,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            horizon=horizon,
            quantiles=quantiles,
            dropout=dropout,
        )
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection for residual connection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Variational dropout between layers
        self.var_dropout = VariationalDropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Output dimension
        fc_input_size = hidden_size * self.num_directions

        # Point forecast head
        self.fc_out = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

        # Quantile heads
        self.quantile_heads = nn.ModuleDict({
            f"q{int(q*100)}": nn.Sequential(
                nn.Linear(fc_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, horizon),
            )
            for q in self.quantiles
        })

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        """Forward pass through LSTM.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            ForecastOutput with point forecast and quantile estimates.
        """
        batch_size = x.size(0)

        # Project input to hidden_size for residual connection
        x_proj = self.input_proj(x)  # (batch, seq_len, hidden_size)

        # Apply variational dropout
        x_proj = self.var_dropout(x_proj)

        # LSTM forward
        lstm_out, (h_n, _) = self.lstm(x_proj)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)

        # Residual connection with last timestep
        if not self.bidirectional:
            features = lstm_out[:, -1, :] + x_proj[:, -1, :]
        else:
            features = lstm_out[:, -1, :]

        # Layer normalization
        features = self.layer_norm(features)  # (batch, hidden_size * num_directions)

        # Point forecast
        point_forecast = self.fc_out(features)  # (batch, horizon)

        # Quantile forecasts
        quantile_preds: dict[float, torch.Tensor] = {}
        for q in self.quantiles:
            quantile_preds[q] = self.quantile_heads[f"q{int(q*100)}"](features)

        return ForecastOutput(
            point_forecast=point_forecast,
            quantiles=quantile_preds,
            metadata={"num_layers": self.num_layers},
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure Adam optimizer with CosineAnnealingLR.

        Returns:
            Dict with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute combined MSE + quantile loss.

        Args:
            batch: Tuple of (input, target) tensors.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        output = self.forward(x)

        # MSE loss for point forecast
        mse_loss = F.mse_loss(output.point_forecast, y)

        # Quantile (pinball) loss
        quantile_loss = torch.tensor(0.0, device=x.device)
        for q, q_pred in output.quantiles.items():
            errors = y - q_pred
            quantile_loss = quantile_loss + torch.mean(
                torch.max(q * errors, (q - 1) * errors)
            )
        quantile_loss = quantile_loss / len(output.quantiles) if output.quantiles else quantile_loss

        return mse_loss + 0.5 * quantile_loss
