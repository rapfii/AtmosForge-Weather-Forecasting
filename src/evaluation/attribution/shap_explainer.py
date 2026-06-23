"""SHAP-based temporal feature importance analysis.

Uses DeepExplainer for LSTM/GRU/CNN and KernelExplainer for others.
Generates temporal importance plots showing which features matter
at which timesteps.

Reference: Lundberg & Lee (2017). A Unified Approach to Interpreting
Model Predictions. NeurIPS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.models.base import BaseForecaster
from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.evaluation.shap")


class SHAPExplainer:
    """SHAP-based feature attribution for time series models.

    Computes temporal feature importance using SHAP values,
    showing which features at which timesteps most influence
    the forecast.

    Args:
        model: Trained BaseForecaster model.
        background_data: Background dataset for SHAP (numpy array).
        n_background: Number of background samples (default: 100).
        feature_names: List of feature names.
        device: Compute device.

    Example:
        >>> explainer = SHAPExplainer(model, background_data)
        >>> shap_values = explainer.explain(test_data)
        >>> explainer.plot_temporal_importance(shap_values)
    """

    def __init__(
        self,
        model: BaseForecaster,
        background_data: np.ndarray[Any, np.dtype[np.floating[Any]]],
        n_background: int = 100,
        feature_names: list[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.feature_names = feature_names
        self.n_background = min(n_background, len(background_data))

        # Sample background data
        indices = np.random.choice(
            len(background_data), self.n_background, replace=False
        )
        self.background = background_data[indices]

        self.model.eval()

    def _model_predict(self, x: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Wrapper for model prediction compatible with SHAP.

        Args:
            x: Input array of shape (n_samples, seq_len, n_features).

        Returns:
            Predictions array.
        """
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            output = self.model(x_tensor)
            return output.point_forecast.cpu().numpy()

    def explain(
        self,
        test_data: np.ndarray[Any, np.dtype[np.floating[Any]]],
        method: str = "auto",
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Compute SHAP values for test data.

        Args:
            test_data: Test samples of shape (n_samples, seq_len, n_features).
            method: 'deep' for DeepExplainer, 'kernel' for KernelExplainer,
                'auto' to pick based on model type.

        Returns:
            SHAP values array of shape (n_samples, seq_len, n_features).
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed. Run: pip install shap")
            raise

        if method == "auto":
            model_type = self.model.__class__.__name__.lower()
            if any(name in model_type for name in ["lstm", "gru", "cnn"]):
                method = "deep"
            else:
                method = "kernel"

        logger.info(f"Computing SHAP values using {method} explainer...")

        if method == "deep":
            try:
                background_tensor = torch.from_numpy(self.background).float().to(self.device)
                explainer = shap.DeepExplainer(self.model, background_tensor)
                test_tensor = torch.from_numpy(test_data).float().to(self.device)
                shap_values = explainer.shap_values(test_tensor)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                return np.array(shap_values)
            except Exception as e:
                logger.warning(f"DeepExplainer failed ({e}), falling back to Kernel")
                method = "kernel"

        if method == "kernel":
            # Flatten for KernelExplainer
            n_samples, seq_len, n_features = test_data.shape
            bg_flat = self.background.reshape(self.n_background, -1)
            test_flat = test_data.reshape(n_samples, -1)

            def flat_predict(x_flat: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
                x_3d = x_flat.reshape(-1, seq_len, n_features)
                return self._model_predict(x_3d).mean(axis=1, keepdims=True)

            explainer = shap.KernelExplainer(flat_predict, bg_flat)
            shap_values_flat = explainer.shap_values(
                test_flat, nsamples=200, silent=True
            )

            if isinstance(shap_values_flat, list):
                shap_values_flat = shap_values_flat[0]

            return np.array(shap_values_flat).reshape(n_samples, seq_len, n_features)

        raise ValueError(f"Unknown method: {method}")

    def plot_temporal_importance(
        self,
        shap_values: np.ndarray[Any, np.dtype[np.floating[Any]]],
        save_path: str | Path | None = None,
        top_k_features: int = 5,
        figsize: tuple[int, int] = (14, 6),
    ) -> None:
        """Plot temporal feature importance heatmap.

        Shows which features at which timesteps most influence
        the forecast, averaged across samples.

        Args:
            shap_values: SHAP values of shape (n_samples, seq_len, n_features).
            save_path: Path to save the plot.
            top_k_features: Number of top features to show.
            figsize: Figure size.
        """
        # Average absolute SHAP values across samples
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # (seq_len, n_features)

        # Get top-k features by total importance
        feature_importance = mean_abs_shap.sum(axis=0)
        top_indices = np.argsort(feature_importance)[-top_k_features:][::-1]

        # Feature names
        if self.feature_names:
            feature_labels = [self.feature_names[i] for i in top_indices]
        else:
            feature_labels = [f"Feature {i}" for i in top_indices]

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        data = mean_abs_shap[:, top_indices].T  # (n_features, seq_len)

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title("Temporal Feature Importance (|SHAP|)", fontsize=14)
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels, fontsize=10)

        plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP plot saved to {save_path}")

        plt.close()

    def compute_and_save(
        self,
        test_data: np.ndarray[Any, np.dtype[np.floating[Any]]],
        model_name: str,
        dataset_name: str,
        horizon: int,
        results_dir: str | Path = "results/shap",
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Convenience method: compute SHAP and save plot.

        Args:
            test_data: Test samples.
            model_name: Name of the model.
            dataset_name: Name of the dataset.
            horizon: Forecast horizon.
            results_dir: Directory for saving results.

        Returns:
            SHAP values array.
        """
        shap_values = self.explain(test_data)

        save_path = Path(results_dir) / f"{model_name}_{dataset_name}_{horizon}h.png"
        self.plot_temporal_importance(shap_values, save_path=save_path)

        return shap_values
