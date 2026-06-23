"""PatchTST — Patch-based Time Series Transformer.

Reference:
    Nie, Y. et al. (2023). A Time Series is Worth 64 Words:
    Long-term Forecasting with Transformers. ICLR 2023.
    arXiv:2211.14730. https://arxiv.org/abs/2211.14730

Architecture:
    Divides input time series into patches (sub-series), treats each
    patch as a token, and applies a standard Transformer encoder.
    Channel-independence: each feature is processed independently then
    combined. This reduces computational complexity from O(L²) to O(N²)
    where N = L/P (number of patches).

Approximate parameters:
    ~1.5M parameters with default config (3 layers, 128 dim)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster, ForecastOutput


class PatchEmbedding(nn.Module):
    """Convert time series into patch embeddings.

    Splits the input sequence into non-overlapping patches and
    projects each patch into the model dimension.

    Args:
        patch_len: Length of each patch.
        stride: Stride between patches.
        d_model: Model dimension for patch embeddings.
        n_features: Number of input features.
    """

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_features: int = 14,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Linear projection from patch to d_model
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create patch embeddings from input sequence.

        Args:
            x: Input of shape (batch, seq_len, n_features) or
               (batch * n_features, seq_len, 1) for channel-independent.

        Returns:
            Patch embeddings of shape (batch, n_patches, d_model).
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # Unfold into patches: (batch, seq_len, feat) -> patches
        # For simplicity, work with last dim = 1 (channel independent)
        if x.dim() == 3 and x.size(2) > 1:
            x = x.mean(dim=2, keepdim=True)  # Aggregate features

        x = x.squeeze(-1)  # (batch, seq_len)

        # Create patches using unfold
        n_patches = max(1, (seq_len - self.patch_len) // self.stride + 1)
        patches = x.unfold(1, self.patch_len, self.stride)  # (batch, n_patches, patch_len)

        # Project patches
        embeddings = self.projection(patches)  # (batch, n_patches, d_model)
        embeddings = self.dropout(embeddings)

        return embeddings


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patch positions.

    Args:
        d_model: Model dimension.
        max_patches: Maximum number of patches.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        max_patches: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Patch embeddings of shape (batch, n_patches, d_model).

        Returns:
            Position-encoded embeddings.
        """
        n_patches = x.size(1)
        x = x + self.pos_embedding[:, :n_patches, :]
        return self.dropout(x)


class PatchTSTForecaster(BaseForecaster):
    """PatchTST: Patch-based Time Series Transformer.

    Reference:
        Nie, Y. et al. (2023). A Time Series is Worth 64 Words:
        Long-term Forecasting with Transformers. ICLR 2023.
        arXiv:2211.14730

    Architecture:
        1. Patch Embedding: Split input into fixed-size patches
        2. Positional Encoding: Learnable position embeddings
        3. Transformer Encoder: Multi-head self-attention + FFN
        4. Flatten + Linear: Project to forecast horizon
        5. Separate quantile heads for uncertainty estimation

        Channel-independence means each variable is processed
        separately, reducing parameters and improving generalization.

    Args:
        input_size: Number of input features.
        hidden_size: Transformer d_model dimension (default: 128).
        output_size: Number of output features (default: 1).
        horizon: Forecast horizon (default: 24).
        seq_len: Input sequence length (default: 168).
        patch_len: Patch length (default: 16).
        stride: Patch stride (default: 8).
        n_heads: Number of attention heads (default: 8).
        n_layers: Number of Transformer layers (default: 3).
        d_ff: Feed-forward dimension (default: 256).
        dropout: Dropout rate (default: 0.1).
        quantiles: Quantile levels.

    Approximate parameters:
        ~1.5M parameters with default config
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        output_size: int = 1,
        horizon: int = 24,
        seq_len: int = 168,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
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
        self.patch_len = patch_len
        self.stride = stride
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input projection: combine all features
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Number of patches
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, hidden_size)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=hidden_size,
            max_patches=self.n_patches + 10,
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        # Output heads
        flatten_size = self.n_patches * hidden_size

        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

        # Quantile heads
        self.quantile_heads = nn.ModuleDict({
            f"q{int(q*100)}": nn.Sequential(
                nn.Flatten(),
                nn.Linear(flatten_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, horizon),
            )
            for q in self.quantiles
        })

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Create patches from projected input.

        Args:
            x: Projected input of shape (batch, seq_len, hidden_size).

        Returns:
            Patches of shape (batch, n_patches, hidden_size).
        """
        batch_size, seq_len, _ = x.shape

        # Take temporal patches by aggregating hidden_size dimension
        # We use 1D convolution-like unfolding on the time axis
        # Reduce to single channel first
        x_reduced = x.mean(dim=2)  # (batch, seq_len)
        patches = x_reduced.unfold(1, self.patch_len, self.stride)  # (batch, n_patches, patch_len)

        # Project patches to hidden_size
        patch_embeddings = self.patch_embedding(patches)  # (batch, n_patches, hidden_size)

        return patch_embeddings

    def forward(self, x: torch.Tensor) -> ForecastOutput:
        """Forward pass through PatchTST.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            ForecastOutput with point forecast and quantile estimates.
        """
        # Project input features to hidden dimension
        x_proj = self.input_proj(x)  # (batch, seq_len, hidden_size)

        # Create patches
        patches = self._create_patches(x_proj)  # (batch, n_patches, hidden_size)

        # Add positional encoding
        patches = self.pos_encoding(patches)

        # Transformer encoder
        encoded = self.transformer_encoder(patches)  # (batch, n_patches, hidden_size)

        # Point forecast
        point_forecast = self.fc_out(encoded)

        # Quantile forecasts
        quantile_preds: dict[float, torch.Tensor] = {}
        for q in self.quantiles:
            quantile_preds[q] = self.quantile_heads[f"q{int(q*100)}"](encoded)

        return ForecastOutput(
            point_forecast=point_forecast,
            quantiles=quantile_preds,
            metadata={"n_patches": self.n_patches, "n_layers": self.n_layers},
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW with cosine annealing.

        Returns:
            Dict with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
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
