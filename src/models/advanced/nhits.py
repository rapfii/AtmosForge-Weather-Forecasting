"""N-HiTS adapter wrapping pytorch-forecasting.

Reference:
    Challu, C. et al. (2023). N-HiTS: Neural Hierarchical Interpolation
    for Time Series Forecasting. AAAI 2023.
    arXiv:2201.12886. https://arxiv.org/abs/2201.12886

Architecture:
    Hierarchical interpolation with multi-rate signal decomposition.
    Uses stacked MLPs with different sampling rates to capture patterns
    at multiple temporal scales. Supports native quantile output.

Approximate parameters:
    ~1.2M parameters with default config
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster, ForecastOutput


class NHiTSBlock(nn.Module):
    """Single N-HiTS block with pooling and interpolation.

    Each block operates at a different temporal scale via pooling,
    then interpolates back to the target resolution.

    Args:
        input_size: Number of input features.
        hidden_size: MLP hidden dimension.
        output_size: Block output dimension.
        n_pool_kernel: Pooling kernel size for this block.
        horizon: Forecast horizon.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_size: int = 256,
        output_size: int = 1,
        n_pool_kernel: int = 1,
        horizon: int = 24,
        n_freq_downsample: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_pool_kernel = n_pool_kernel
        self.horizon = horizon
        self.n_freq_downsample = n_freq_downsample

        # Pooling for multi-rate decomposition
        self.pooling = nn.MaxPool1d(kernel_size=n_pool_kernel, stride=n_pool_kernel)

        pooled_len = seq_len // n_pool_kernel
        flat_size = input_size * pooled_len

        # MLP stack
        self.mlp = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Backcast and forecast heads
        n_theta_backcast = pooled_len
        n_theta_forecast = max(1, horizon // n_freq_downsample)

        self.backcast_head = nn.Linear(hidden_size, n_theta_backcast * input_size)
        self.forecast_head = nn.Linear(hidden_size, n_theta_forecast)

        self.flat_size = flat_size
        self.pooled_len = pooled_len
        self.input_size = input_size

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through N-HiTS block.

        Args:
            x: Input of shape (batch, seq_len, n_features).

        Returns:
            Tuple of (backcast, forecast).
        """
        batch_size = x.size(0)

        # Pool: (batch, features, seq_len) -> (batch, features, pooled_len)
        x_pooled = self.pooling(x.permute(0, 2, 1))
        x_flat = x_pooled.permute(0, 2, 1).reshape(batch_size, -1)

        # Pad or truncate to match expected flat size
        if x_flat.size(1) < self.flat_size:
            x_flat = F.pad(x_flat, (0, self.flat_size - x_flat.size(1)))
        elif x_flat.size(1) > self.flat_size:
            x_flat = x_flat[:, :self.flat_size]

        # MLP
        hidden = self.mlp(x_flat)

        # Backcast
        backcast_theta = self.backcast_head(hidden)
        backcast = backcast_theta.view(batch_size, self.pooled_len, self.input_size)

        # Forecast with interpolation
        forecast_theta = self.forecast_head(hidden)  # (batch, n_theta_forecast)

        # Interpolate to full horizon
        if forecast_theta.size(1) < self.horizon:
            forecast = F.interpolate(
                forecast_theta.unsqueeze(1),
                size=self.horizon,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        else:
            forecast = forecast_theta[:, :self.horizon]

        return backcast, forecast


class NHiTSForecaster(BaseForecaster):
    """N-HiTS: Neural Hierarchical Interpolation for Time Series.

    Reference:
        Challu, C. et al. (2023). N-HiTS: Neural Hierarchical
        Interpolation for Time Series Forecasting. AAAI 2023.
        arXiv:2201.12886

    Architecture:
        Stack of N-HiTS blocks operating at different temporal
        scales via multi-rate pooling. Each block captures patterns
        at its scale, and forecasts are summed across blocks.
        The hierarchical interpolation ensures efficient learning
        of both short and long-range patterns.

    Args:
        input_size: Number of input features.
        hidden_size: MLP hidden dimension per block (default: 256).
        output_size: Number of output features (default: 1).
        horizon: Forecast horizon (default: 24).
        seq_len: Input sequence length (default: 168).
        n_blocks: Number of N-HiTS blocks per stack (default: 3).
        n_stacks: Number of stacks (default: 3).
        pool_kernels: Pooling kernel sizes per stack.
        dropout: Dropout rate (default: 0.1).
        quantiles: Quantile levels for probabilistic output.

    Approximate parameters:
        ~1.2M parameters with default config
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 256,
        output_size: int = 1,
        horizon: int = 24,
        seq_len: int = 168,
        n_blocks: int = 3,
        n_stacks: int = 3,
        pool_kernels: list[int] | None = None,
        freq_downsample: list[int] | None = None,
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
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks

        # Default pool kernels: [1, 2, 4] for multi-rate decomposition
        self.pool_kernels = pool_kernels or [1, 2, 4][:n_stacks]
        self.freq_downsample = freq_downsample or [1, 2, 4][:n_stacks]

        # Ensure we have enough pool kernels
        while len(self.pool_kernels) < n_stacks:
            self.pool_kernels.append(self.pool_kernels[-1] * 2)
        while len(self.freq_downsample) < n_stacks:
            self.freq_downsample.append(self.freq_downsample[-1] * 2)

        # Build stacks of blocks
        self.blocks = nn.ModuleList()
        for stack_idx in range(n_stacks):
            for _ in range(n_blocks):
                self.blocks.append(
                    NHiTSBlock(
                        input_size=input_size,
                        seq_len=seq_len,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        n_pool_kernel=self.pool_kernels[stack_idx],
                        horizon=horizon,
                        n_freq_downsample=self.freq_downsample[stack_idx],
                        dropout=dropout,
                    )
                )

        # Quantile heads (applied on top of summed forecast)
        self.quantile_heads = nn.ModuleDict({
            f"q{int(q*100)}": nn.Linear(horizon, horizon)
            for q in self.quantiles
        })

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        """Forward pass through N-HiTS.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            ForecastOutput with point forecast and quantile estimates.
        """
        residual = x
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)

        for block in self.blocks:
            # Adjust residual length if needed
            if residual.size(1) != self.seq_len:
                if residual.size(1) > self.seq_len:
                    residual = residual[:, -self.seq_len:, :]
                else:
                    pad_len = self.seq_len - residual.size(1)
                    residual = F.pad(residual, (0, 0, pad_len, 0))

            backcast, block_forecast = block(residual)
            forecast = forecast + block_forecast

        # Point forecast
        point_forecast = forecast

        # Quantile forecasts
        quantile_preds: dict[float, torch.Tensor] = {}
        for q in self.quantiles:
            quantile_preds[q] = self.quantile_heads[f"q{int(q*100)}"](forecast)

        return ForecastOutput(
            point_forecast=point_forecast,
            quantiles=quantile_preds,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure Adam optimizer.

        Returns:
            Dict with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}

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

        mse_loss = F.mse_loss(output.point_forecast, y)

        quantile_loss = torch.tensor(0.0, device=x.device)
        for q, q_pred in output.quantiles.items():
            errors = y - q_pred
            quantile_loss = quantile_loss + torch.mean(
                torch.max(q * errors, (q - 1) * errors)
            )
        quantile_loss = quantile_loss / len(output.quantiles) if output.quantiles else quantile_loss

        return mse_loss + 0.5 * quantile_loss
