"""Dilated Temporal Convolutional Network (TCN-style CNN-1D).

Reference:
    Bai, S. et al. (2018). An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling.
    arXiv:1803.01271. https://arxiv.org/abs/1803.01271

Architecture:
    Stacked dilated causal 1D convolutions with residual connections.
    The receptive field grows exponentially with depth via dilation,
    allowing the model to capture long-range temporal dependencies
    without the vanishing gradient issues of RNNs.

Approximate parameters:
    ~0.3M parameters with default config (4 layers, 64 filters)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster, ForecastOutput


class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation.

    Applies left-padding to ensure output depends only on current
    and past inputs (no future information leakage).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal padding.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor of shape (batch, channels, seq_len).
        """
        out = self.conv(x)
        # Remove future padding to maintain causal property
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal conv + residual.

    Contains two causal convolutions with weight normalization,
    dropout, and a residual connection.

    Args:
        in_channels: Input channel size.
        out_channels: Output channel size.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.utils.weight_norm(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation).conv
        )
        self.conv2 = nn.utils.weight_norm(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation).conv
        )
        self.padding1 = (kernel_size - 1) * dilation
        self.padding2 = (kernel_size - 1) * dilation

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection (1x1 conv if channel mismatch)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output with residual of shape (batch, channels, seq_len).
        """
        # First conv block
        out = self.conv1(x)
        if self.padding1 > 0:
            out = out[:, :, :-self.padding1]
        out = self.relu(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        if self.padding2 > 0:
            out = out[:, :, :-self.padding2]
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual
        res = self.residual(x)
        return self.relu(out + res)


class CNN1DForecaster(BaseForecaster):
    """Dilated Temporal Convolutional Network (TCN) for time series forecasting.

    Reference:
        Bai, S. et al. (2018). An Empirical Evaluation of Generic
        Convolutional and Recurrent Networks for Sequence Modeling.
        arXiv:1803.01271

    Architecture:
        Stacked temporal blocks with exponentially growing dilation
        factors [1, 2, 4, 8, ...]. Each block contains two causal
        1D convolutions with residual connections. Final output
        is projected through a linear head for point forecast and
        separate quantile heads.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Number of filters per conv layer (default: 64).
        output_size: Number of output features (default: 1).
        horizon: Forecast horizon (default: 24).
        num_layers: Number of temporal blocks (default: 4).
        kernel_size: Convolution kernel size (default: 3).
        dropout: Dropout rate (default: 0.1).
        quantiles: Quantile levels for probabilistic output.

    Approximate parameters:
        ~0.3M parameters with default config
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        output_size: int = 1,
        horizon: int = 24,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
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
        self.kernel_size = kernel_size

        # Build temporal blocks with exponential dilation
        layers: list[nn.Module] = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else hidden_size
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)

        # Output projection: point forecast
        self.fc_out = nn.Linear(hidden_size, horizon)

        # Quantile heads (separate linear layers per quantile)
        self.quantile_heads = nn.ModuleDict({
            f"q{int(q*100)}": nn.Linear(hidden_size, horizon)
            for q in self.quantiles
        })

        # Receptive field info
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_layers - 1)

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        """Forward pass through TCN.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            ForecastOutput with point forecast and quantile estimates.
        """
        # TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Pass through temporal blocks
        tcn_out = self.tcn(x)  # (batch, hidden_size, seq_len)

        # Take the last timestep's representation
        features = tcn_out[:, :, -1]  # (batch, hidden_size)

        # Point forecast
        point_forecast = self.fc_out(features)  # (batch, horizon)

        # Quantile forecasts
        quantile_preds: dict[float, torch.Tensor] = {}
        for q in self.quantiles:
            quantile_preds[q] = self.quantile_heads[f"q{int(q*100)}"](features)

        return ForecastOutput(
            point_forecast=point_forecast,
            quantiles=quantile_preds,
            metadata={"receptive_field": self.receptive_field},
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure Adam optimizer with ReduceLROnPlateau.

        Returns:
            Dict with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
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
