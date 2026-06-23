"""Temporal Fusion Transformer (TFT) adapter.

Reference:
    Lim, B. et al. (2021). Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting. International Journal of
    Forecasting. arXiv:1912.09363. https://arxiv.org/abs/1912.09363

Architecture:
    Multi-horizon forecasting with variable selection, gating mechanisms,
    and interpretable multi-head attention. Supports both static and
    temporal inputs with native quantile output.

    This implementation provides a standalone TFT that follows the
    BaseForecaster interface, with attention-based temporal processing.

Approximate parameters:
    ~2.0M parameters with default config (3 layers, 128 hidden)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster, ForecastOutput


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for information flow control.

    Args:
        input_size: Input dimension.
        hidden_size: Output dimension.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GLU.

        Args:
            x: Input tensor.

        Returns:
            Gated output tensor.
        """
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) — core building block of TFT.

    Processes input through dense layers with ELU activation,
    gated linear unit, and layer normalization with residual.

    Args:
        input_size: Input dimension.
        hidden_size: Hidden dimension.
        output_size: Output dimension.
        dropout: Dropout rate.
        context_size: Optional context vector dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        context_size: int | None = None,
    ) -> None:
        super().__init__()
        output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

        # Residual connection (project if sizes differ)
        self.residual = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else nn.Identity()
        )

        # Optional context
        self.context_proj = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size is not None
            else None
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through GRN.

        Args:
            x: Input tensor.
            context: Optional context vector.

        Returns:
            Processed tensor.
        """
        residual = self.residual(x)

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        gated = self.glu(hidden)
        return self.layer_norm(gated + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for input feature weighting.

    Learns to weight the importance of each input variable,
    providing interpretability about which features matter.

    Args:
        input_size: Number of input features.
        hidden_size: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(input_size)
        ])

        # Variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_size * hidden_size, hidden_size, input_size, dropout
        )

        self.softmax = nn.Softmax(dim=-1)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with variable selection.

        Args:
            x: Input of shape (batch, seq_len, n_features).

        Returns:
            Tuple of (selected_features, variable_weights).
        """
        batch_size, seq_len, _ = x.shape

        # Process each variable through its GRN
        var_outputs = []
        for i in range(self.input_size):
            var_input = x[:, :, i:i+1]  # (batch, seq_len, 1)
            var_out = self.var_grns[i](var_input)  # (batch, seq_len, hidden_size)
            var_outputs.append(var_out)

        # Concatenate and compute selection weights
        var_concat = torch.cat(var_outputs, dim=-1)  # (batch, seq_len, input_size * hidden)
        weights = self.selection_grn(var_concat)  # (batch, seq_len, input_size)
        weights = self.softmax(weights)  # Normalize

        # Apply variable selection
        var_stack = torch.stack(var_outputs, dim=-1)  # (batch, seq_len, hidden, input_size)
        selected = (var_stack * weights.unsqueeze(2)).sum(dim=-1)  # (batch, seq_len, hidden)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention for TFT.

    Modified multi-head attention where attention weights are
    shared across heads for interpretability.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through attention.

        Args:
            query: Query tensor (batch, seq_len, d_model).
            key: Key tensor.
            value: Value tensor.
            mask: Optional attention mask.

        Returns:
            Tuple of (output, attention_weights).
        """
        batch_size = query.size(0)

        # Project and reshape
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)

        # Average attention weights across heads for interpretability
        avg_weights = attn_weights.mean(dim=1)

        return output, avg_weights


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer for interpretable multi-horizon forecasting.

    Reference:
        Lim, B. et al. (2021). Temporal Fusion Transformers for
        Interpretable Multi-horizon Time Series Forecasting.
        International Journal of Forecasting.
        arXiv:1912.09363

    Architecture:
        1. Variable Selection Network: Learns feature importance
        2. LSTM Encoder: Captures temporal patterns
        3. Interpretable Multi-Head Attention: Self-attention with
           interpretable weights
        4. Gated Residual Networks: Information flow control
        5. Quantile Output: Native multi-quantile prediction

    Args:
        input_size: Number of input features.
        hidden_size: Model hidden dimension (default: 128).
        output_size: Number of output features (default: 1).
        horizon: Forecast horizon (default: 24).
        n_heads: Number of attention heads (default: 4).
        n_layers: Number of LSTM layers (default: 2).
        dropout: Dropout rate (default: 0.1).
        quantiles: Quantile levels.

    Approximate parameters:
        ~2.0M parameters with default config
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        output_size: int = 1,
        horizon: int = 24,
        n_heads: int = 4,
        n_layers: int = 2,
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
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Variable selection
        self.vsn = VariableSelectionNetwork(input_size, hidden_size, dropout)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Post-LSTM GRN + layer norm
        self.post_lstm_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)

        # Interpretable multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_size, n_heads=n_heads, dropout=dropout
        )

        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # Output projection for each quantile
        self.quantile_projections = nn.ModuleDict({
            f"q{int(q*100)}": nn.Linear(hidden_size, horizon)
            for q in self.quantiles
        })

        # Point forecast
        self.point_projection = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        """Forward pass through TFT.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            ForecastOutput with point forecast, quantiles, and attention.
        """
        # Variable selection
        selected, var_weights = self.vsn(x)  # (batch, seq_len, hidden)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected)  # (batch, seq_len, hidden)

        # Post-LSTM processing with residual
        lstm_processed = self.post_lstm_grn(lstm_out)  # (batch, seq_len, hidden)

        # Self-attention
        attn_out, attn_weights = self.attention(
            lstm_processed, lstm_processed, lstm_processed
        )

        # Post-attention with residual and layer norm
        attn_processed = self.post_attn_grn(attn_out)
        attn_final = self.post_attn_norm(attn_processed + lstm_processed)

        # Use last timestep for forecast
        features = attn_final[:, -1, :]  # (batch, hidden)

        # Point forecast
        point_forecast = self.point_projection(features)

        # Quantile forecasts
        quantile_preds: dict[float, torch.Tensor] = {}
        for q in self.quantiles:
            quantile_preds[q] = self.quantile_projections[f"q{int(q*100)}"](features)

        return ForecastOutput(
            point_forecast=point_forecast,
            quantiles=quantile_preds,
            metadata={
                "attention_weights": attn_weights.detach(),
                "variable_weights": var_weights.detach(),
            },
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW with OneCycleLR.

        Returns:
            Dict with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute combined quantile loss (TFT native loss).

        Args:
            batch: Tuple of (input, target) tensors.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        output = self.forward(x)

        # Primary: quantile loss (TFT's native objective)
        total_loss = torch.tensor(0.0, device=x.device)
        for q, q_pred in output.quantiles.items():
            errors = y - q_pred
            total_loss = total_loss + torch.mean(
                torch.max(q * errors, (q - 1) * errors)
            )

        # Add MSE for point forecast stability
        mse_loss = F.mse_loss(output.point_forecast, y)
        total_loss = total_loss / len(output.quantiles) + 0.5 * mse_loss

        return total_loss
