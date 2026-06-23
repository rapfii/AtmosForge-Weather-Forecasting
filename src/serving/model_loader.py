"""Model loader for inference — loads from checkpoint or MLflow registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.base import BaseForecaster
from src.training.trainer import MODEL_REGISTRY, register_models
from src.utils.logger import setup_logger
from src.utils.seed import get_device

logger = setup_logger("atmosforge.serving.model_loader")


class ModelLoader:
    """Load trained models for inference.

    Supports loading from:
    1. Local checkpoint files (.pt)
    2. MLflow model registry

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        mlflow_uri: MLflow tracking URI.
        device: Inference device.

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load("lstm", checkpoint_path="checkpoints/best.pt")
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        mlflow_uri: str = "./mlruns",
        device: torch.device | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.mlflow_uri = mlflow_uri
        self.device = device or get_device()
        self._loaded_models: dict[str, BaseForecaster] = {}

        register_models()

    def load_from_checkpoint(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        model_config: dict[str, Any] | None = None,
    ) -> BaseForecaster:
        """Load model from a local checkpoint file.

        Args:
            model_name: Model name (cnn, lstm, gru, etc.).
            checkpoint_path: Path to .pt checkpoint file.
            model_config: Model constructor parameters.

        Returns:
            Loaded model in eval mode on the target device.
        """
        config = model_config or {}
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model = MODEL_REGISTRY[model_name](**config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        self._loaded_models[model_name] = model
        logger.info(f"Loaded {model_name} from {checkpoint_path}")
        return model

    def load_from_mlflow(
        self, model_name: str, run_id: str | None = None
    ) -> BaseForecaster | None:
        """Load model from MLflow registry.

        Args:
            model_name: Model name.
            run_id: Specific MLflow run ID (latest if None).

        Returns:
            Loaded model or None if not found.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_uri)

            if run_id:
                artifact_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id
                )
            else:
                # Search for latest successful run
                experiment = mlflow.get_experiment_by_name("atmosforge")
                if experiment is None:
                    logger.warning("No MLflow experiment found")
                    return None

                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"params.model = '{model_name}' AND status = 'FINISHED'",
                    order_by=["metrics.best_val_loss ASC"],
                    max_results=1,
                )

                if runs.empty:
                    logger.warning(f"No runs found for model {model_name}")
                    return None

                run_id = runs.iloc[0]["run_id"]
                logger.info(f"Loading best run: {run_id}")

                # Find checkpoint artifact
                artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
                checkpoint_artifacts = [
                    a for a in artifacts if a.path.endswith(".pt")
                ]

                if not checkpoint_artifacts:
                    logger.warning(f"No checkpoint found for run {run_id}")
                    return None

                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=checkpoint_artifacts[0].path,
                )

                return self.load_from_checkpoint(model_name, local_path)

        except ImportError:
            logger.error("MLflow not installed")
        except Exception as e:
            logger.error(f"Failed to load from MLflow: {e}")

        return None

    def get_loaded_model(self, model_name: str) -> BaseForecaster | None:
        """Get a previously loaded model.

        Args:
            model_name: Model name.

        Returns:
            Model if loaded, None otherwise.
        """
        return self._loaded_models.get(model_name)

    def list_available_checkpoints(self) -> list[dict[str, Any]]:
        """List available checkpoint files.

        Returns:
            List of checkpoint info dicts.
        """
        checkpoints = []
        if self.checkpoint_dir.exists():
            for pt_file in self.checkpoint_dir.glob("*.pt"):
                checkpoints.append({
                    "filename": pt_file.name,
                    "path": str(pt_file),
                    "size_mb": pt_file.stat().st_size / 1e6,
                })
        return checkpoints
