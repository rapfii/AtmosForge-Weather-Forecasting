"""GRU forecaster with residual connections and variational dropout.

Reference:
    Cho, K. et al. (2014). Learning Phrase Representations using RNN
    Encoder-Decoder for Statistical Machine Translation.
    arXiv:1406.1078. https://arxiv.org/abs/1406.1078

Architecture:
    Multi-layer GRU with residual connections and variational dropout.
    More parameter-efficient than LSTM (3 gates vs 4) while achieving
    comparable performance on many time series tasks.

Approximate parameters:
    ~0.4M parameters with default config (2 layers, 128 hidden)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster, ForecastOutput
from src.models.baselines.lstm import VariationalDropout


class GRUForecaster(BaseForecaster):
    """GRU with residual connections for time series forecasting.

    Reference:
        Cho, K. et al. (2014). Learning Phrase Representations using
        RNN Encoder-Decoder for Statistical Machine Translation.
        arXiv:1406.1078

    Architecture:
        Multi-layer GRU with:
        - Residual connections between layers
        - Variational dropout (consistent mask across timesteps)
        - Layer normalization on hidden states
        - Point forecast head + separate quantile heads

        GRU uses only 3 gates (reset, update, new) compared to LSTM's
        4 gates, making it ~25% more parameter-efficient.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: GRU hidden state dimension (default: 128).
        output_size: Number of output features (default: 1).
        horizon: Forecast horizon (default: 24).
        num_layers: Number of GRU layers (default: 2).
        dropout: Dropout rate (default: 0.1).
        bidirectional: Use bidirectional GRU (default: False).
        quantiles: Quantile levels for probabilistic output.

    Approximate parameters:
        ~0.4M parameters with default config
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

        # Input projection for residual
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Variational dropout
        self.var_dropout = VariationalDropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

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
        """Forward pass through GRU.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            ForecastOutput with point forecast and quantile estimates.
        """
        # Project input for residual
        x_proj = self.input_proj(x)
        x_proj = self.var_dropout(x_proj)

        # GRU forward
        gru_out, h_n = self.gru(x_proj)

        # Residual connection + layer norm
        if not self.bidirectional:
            features = gru_out[:, -1, :] + x_proj[:, -1, :]
        else:
            features = gru_out[:, -1, :]

        features = self.layer_norm(features)

        # Point forecast
        point_forecast = self.fc_out(features)

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
        """Configure Adam optimizer with ReduceLROnPlateau.

        Returns:
            Dict with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=1e-5
        )
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

        mse_loss = F.mse_loss(output.point_forecast, y)

        quantile_loss = torch.tensor(0.0, device=x.device)
        for q, q_pred in output.quantiles.items():
            errors = y - q_pred
            quantile_loss = quantile_loss + torch.mean(
                torch.max(q * errors, (q - 1) * errors)
            )
        quantile_loss = quantile_loss / len(output.quantiles) if output.quantiles else quantile_loss

        return mse_loss + 0.5 * quantile_loss
