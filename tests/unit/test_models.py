"""Unit tests for model forward pass validation.

Tests that all 6 models can:
1. Accept input of correct shape (batch, seq_len, n_features)
2. Produce ForecastOutput with correct shapes
3. Return quantile predictions when configured
"""

from __future__ import annotations

import pytest
import torch

from src.models.advanced.nhits import NHiTSForecaster
from src.models.advanced.patchtst import PatchTSTForecaster
from src.models.advanced.tft import TFTForecaster
from src.models.base import ForecastOutput
from src.models.baselines.cnn import CNN1DForecaster
from src.models.baselines.gru import GRUForecaster
from src.models.baselines.lstm import LSTMForecaster

# Test configuration
BATCH_SIZE = 4
SEQ_LEN = 168
N_FEATURES = 14
HORIZON = 24
QUANTILES = [0.1, 0.5, 0.9]


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Create dummy input tensor."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)


@pytest.fixture
def dummy_target() -> torch.Tensor:
    """Create dummy target tensor."""
    return torch.randn(BATCH_SIZE, HORIZON)


class TestCNN1D:
    """Tests for CNN-1D (TCN) model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        model = CNN1DForecaster(
            input_size=N_FEATURES, hidden_size=32, horizon=HORIZON, num_layers=2
        )
        output = model(dummy_input)
        assert isinstance(output, ForecastOutput)
        assert output.point_forecast.shape == (BATCH_SIZE, HORIZON)

    def test_quantile_output(self, dummy_input: torch.Tensor) -> None:
        model = CNN1DForecaster(
            input_size=N_FEATURES, hidden_size=32, horizon=HORIZON,
            quantiles=QUANTILES
        )
        output = model(dummy_input)
        assert len(output.quantiles) == 3
        for q in QUANTILES:
            assert q in output.quantiles
            assert output.quantiles[q].shape == (BATCH_SIZE, HORIZON)

    def test_training_step(
        self, dummy_input: torch.Tensor, dummy_target: torch.Tensor
    ) -> None:
        model = CNN1DForecaster(input_size=N_FEATURES, hidden_size=32, horizon=HORIZON)
        loss = model.training_step((dummy_input, dummy_target))
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_parameter_count(self) -> None:
        model = CNN1DForecaster(input_size=N_FEATURES, hidden_size=64, num_layers=4)
        assert model.count_parameters() > 0


class TestLSTM:
    """Tests for LSTM model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        model = LSTMForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON, num_layers=2
        )
        output = model(dummy_input)
        assert output.point_forecast.shape == (BATCH_SIZE, HORIZON)

    def test_quantile_output(self, dummy_input: torch.Tensor) -> None:
        model = LSTMForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            quantiles=QUANTILES
        )
        output = model(dummy_input)
        assert len(output.quantiles) == 3

    def test_training_step(
        self, dummy_input: torch.Tensor, dummy_target: torch.Tensor
    ) -> None:
        model = LSTMForecaster(input_size=N_FEATURES, hidden_size=64, horizon=HORIZON)
        loss = model.training_step((dummy_input, dummy_target))
        assert loss.dim() == 0


class TestGRU:
    """Tests for GRU model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        model = GRUForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON, num_layers=2
        )
        output = model(dummy_input)
        assert output.point_forecast.shape == (BATCH_SIZE, HORIZON)

    def test_quantile_output(self, dummy_input: torch.Tensor) -> None:
        model = GRUForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            quantiles=QUANTILES
        )
        output = model(dummy_input)
        assert len(output.quantiles) == 3

    def test_training_step(
        self, dummy_input: torch.Tensor, dummy_target: torch.Tensor
    ) -> None:
        model = GRUForecaster(input_size=N_FEATURES, hidden_size=64, horizon=HORIZON)
        loss = model.training_step((dummy_input, dummy_target))
        assert loss.dim() == 0


class TestNHiTS:
    """Tests for N-HiTS model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        model = NHiTSForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            seq_len=SEQ_LEN, n_blocks=2, n_stacks=2
        )
        output = model(dummy_input)
        assert output.point_forecast.shape == (BATCH_SIZE, HORIZON)

    def test_quantile_output(self, dummy_input: torch.Tensor) -> None:
        model = NHiTSForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            seq_len=SEQ_LEN, n_blocks=2, n_stacks=2, quantiles=QUANTILES
        )
        output = model(dummy_input)
        assert len(output.quantiles) == 3

    def test_training_step(
        self, dummy_input: torch.Tensor, dummy_target: torch.Tensor
    ) -> None:
        model = NHiTSForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            seq_len=SEQ_LEN, n_blocks=2, n_stacks=2
        )
        loss = model.training_step((dummy_input, dummy_target))
        assert loss.dim() == 0


class TestPatchTST:
    """Tests for PatchTST model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        model = PatchTSTForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            seq_len=SEQ_LEN, n_heads=4, n_layers=2
        )
        output = model(dummy_input)
        assert output.point_forecast.shape == (BATCH_SIZE, HORIZON)

    def test_quantile_output(self, dummy_input: torch.Tensor) -> None:
        model = PatchTSTForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            seq_len=SEQ_LEN, n_heads=4, n_layers=2, quantiles=QUANTILES
        )
        output = model(dummy_input)
        assert len(output.quantiles) == 3

    def test_training_step(
        self, dummy_input: torch.Tensor, dummy_target: torch.Tensor
    ) -> None:
        model = PatchTSTForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            seq_len=SEQ_LEN, n_heads=4, n_layers=2
        )
        loss = model.training_step((dummy_input, dummy_target))
        assert loss.dim() == 0


class TestTFT:
    """Tests for TFT model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        model = TFTForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            n_heads=4, n_layers=2
        )
        output = model(dummy_input)
        assert output.point_forecast.shape == (BATCH_SIZE, HORIZON)

    def test_quantile_output(self, dummy_input: torch.Tensor) -> None:
        model = TFTForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            n_heads=4, n_layers=2, quantiles=QUANTILES
        )
        output = model(dummy_input)
        assert len(output.quantiles) == 3

    def test_attention_weights(self, dummy_input: torch.Tensor) -> None:
        model = TFTForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            n_heads=4, n_layers=2
        )
        output = model(dummy_input)
        assert "attention_weights" in output.metadata
        assert "variable_weights" in output.metadata

    def test_training_step(
        self, dummy_input: torch.Tensor, dummy_target: torch.Tensor
    ) -> None:
        model = TFTForecaster(
            input_size=N_FEATURES, hidden_size=64, horizon=HORIZON,
            n_heads=4, n_layers=2
        )
        loss = model.training_step((dummy_input, dummy_target))
        assert loss.dim() == 0


class TestModelConsistency:
    """Cross-model consistency tests."""

    @pytest.mark.parametrize("model_class,kwargs", [
        (CNN1DForecaster, {"hidden_size": 32, "num_layers": 2}),
        (LSTMForecaster, {"hidden_size": 64, "num_layers": 2}),
        (GRUForecaster, {"hidden_size": 64, "num_layers": 2}),
        (NHiTSForecaster, {"hidden_size": 64, "seq_len": SEQ_LEN, "n_blocks": 2, "n_stacks": 2}),
        (PatchTSTForecaster, {"hidden_size": 64, "seq_len": SEQ_LEN, "n_heads": 4, "n_layers": 2}),
        (TFTForecaster, {"hidden_size": 64, "n_heads": 4, "n_layers": 2}),
    ])
    def test_all_models_same_interface(
        self,
        dummy_input: torch.Tensor,
        model_class: type,
        kwargs: dict[str, int],
    ) -> None:
        """All models should accept same input shape and produce same output shape."""
        model = model_class(
            input_size=N_FEATURES, horizon=HORIZON, quantiles=QUANTILES, **kwargs
        )
        output = model(dummy_input)

        assert isinstance(output, ForecastOutput)
        assert output.point_forecast.shape == (BATCH_SIZE, HORIZON)
        assert len(output.quantiles) == len(QUANTILES)
        for q in QUANTILES:
            assert output.quantiles[q].shape == (BATCH_SIZE, HORIZON)

    def test_all_models_have_configure_optimizers(self) -> None:
        """All models must implement configure_optimizers."""
        for cls in [CNN1DForecaster, LSTMForecaster, GRUForecaster,
                    NHiTSForecaster, PatchTSTForecaster, TFTForecaster]:
            model = cls(input_size=N_FEATURES, hidden_size=32, horizon=HORIZON)
            config = model.configure_optimizers()
            assert "optimizer" in config
