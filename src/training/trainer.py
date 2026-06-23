"""Generic model-agnostic trainer with MLflow tracking.

Supports any model inheriting from BaseForecaster. Features:
- MLflow experiment tracking (params, metrics, artifacts)
- EarlyStopping with configurable patience
- Gradient clipping
- Mixed precision training (AMP) for RTX 4050 efficiency
- Checkpoint save/load for resume training
- LR scheduling (ReduceLROnPlateau, CosineAnnealing, OneCycleLR)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.base import BaseForecaster
from src.training.callbacks.early_stopping import EarlyStopping
from src.utils.logger import setup_logger
from src.utils.seed import get_device, set_seed

logger = setup_logger("atmosforge.training.trainer")

# Model registry for instantiation from config
MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {}


def register_models() -> None:
    """Register all available models."""
    from src.models.advanced.nhits import NHiTSForecaster
    from src.models.advanced.patchtst import PatchTSTForecaster
    from src.models.advanced.tft import TFTForecaster
    from src.models.baselines.cnn import CNN1DForecaster
    from src.models.baselines.gru import GRUForecaster
    from src.models.baselines.lstm import LSTMForecaster

    MODEL_REGISTRY.update({
        "cnn": CNN1DForecaster,
        "lstm": LSTMForecaster,
        "gru": GRUForecaster,
        "nhits": NHiTSForecaster,
        "patchtst": PatchTSTForecaster,
        "tft": TFTForecaster,
    })


class GenericTrainer:
    """Model-agnostic training loop with full MLOps instrumentation.

    Trains any BaseForecaster model using a standardized pipeline with
    MLflow tracking, early stopping, mixed precision, and checkpointing.

    Args:
        model: BaseForecaster model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        max_epochs: Maximum training epochs (default: 100).
        patience: EarlyStopping patience (default: 10).
        grad_clip_norm: Gradient clipping max norm (default: 1.0).
        use_amp: Use Automatic Mixed Precision (default: auto-detect CUDA).
        checkpoint_dir: Directory for saving checkpoints.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        device: Compute device (default: auto-detect).
        seed: Random seed (default: 42).

    Example:
        >>> trainer = GenericTrainer(model, train_loader, val_loader)
        >>> history = trainer.fit()
    """

    def __init__(
        self,
        model: BaseForecaster,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip_norm: float = 1.0,
        use_amp: bool | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        experiment_name: str = "atmosforge",
        run_name: str | None = None,
        device: torch.device | None = None,
        seed: int = 42,
        learning_rate: float = 1e-3,
        scheduler_type: str = "reduce_on_plateau",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.run_name = run_name or f"{model.__class__.__name__}_{int(time.time())}"
        self.seed = seed
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type

        # Device setup (prioritize RTX 4050 GPU)
        self.device = device or get_device()
        logger.info(f"Training device: {self.device}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # AMP setup (great for RTX 4050 with Tensor Cores)
        self.use_amp = use_amp if use_amp is not None else self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Move model to device
        self.model = self.model.to(self.device)

        # Configure optimizer and scheduler
        optim_config = model.configure_optimizers()
        self.optimizer = optim_config["optimizer"]
        self.scheduler = optim_config.get("scheduler")

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def _train_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # Move to device
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(enabled=self.use_amp):
                loss = self.model.training_step((x, y))

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self) -> dict[str, float]:
        """Run validation epoch.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                metrics = self.model.validation_step((x, y))

            total_loss += metrics["val_loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"val_loss": avg_loss}

    def save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """Save training checkpoint.

        Args:
            epoch: Current epoch number.
            val_loss: Current validation loss.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"{self.run_name}_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "val_loss": val_loss,
                "scaler_state_dict": self.scaler.state_dict(),
                "history": self.history,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load training checkpoint to resume.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Epoch number to resume from.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    def fit(self, resume_from: str | Path | None = None) -> dict[str, list[float]]:
        """Train the model with full MLflow tracking.

        Args:
            resume_from: Optional checkpoint path to resume from.

        Returns:
            Training history dictionary.
        """
        # Set seed for reproducibility
        set_seed(self.seed)

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        # MLflow tracking
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name):
            # Log parameters
            mlflow.log_params({
                "model": self.model.__class__.__name__,
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_size,
                "horizon": self.model.horizon,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "grad_clip_norm": self.grad_clip_norm,
                "use_amp": self.use_amp,
                "device": str(self.device),
                "seed": self.seed,
                "total_params": self.model.count_parameters(),
                "batch_size": self.train_loader.batch_size or 0,
                "learning_rate": self.learning_rate,
                "scheduler": self.scheduler_type,
            })

            best_val_loss = float("inf")
            best_checkpoint: Path | None = None

            logger.info(
                f"Starting training: {self.model.__class__.__name__} | "
                f"epochs={self.max_epochs} | device={self.device} | "
                f"AMP={'ON' if self.use_amp else 'OFF'} | "
                f"params={self.model.count_parameters():,}"
            )

            for epoch in range(start_epoch, self.max_epochs):
                epoch_start = time.time()

                # Train
                train_loss = self._train_epoch()

                # Validate
                val_metrics = self._validate_epoch()
                val_loss = val_metrics["val_loss"]

                # LR scheduler step
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                epoch_time = time.time() - epoch_start

                # Record history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["learning_rate"].append(current_lr)

                # MLflow logging
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": current_lr,
                        "epoch_time_s": epoch_time,
                    },
                    step=epoch,
                )

                # Log progress
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs} │ "
                    f"train_loss={train_loss:.6f} │ "
                    f"val_loss={val_loss:.6f} │ "
                    f"lr={current_lr:.2e} │ "
                    f"time={epoch_time:.1f}s"
                )

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = self.save_checkpoint(epoch, val_loss)
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                    logger.info(f"  ★ New best: val_loss={best_val_loss:.6f}")

                # Early stopping check
                if self.early_stopping.step(val_loss):
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(patience={self.patience})"
                    )
                    break

            # Log best model artifact to MLflow
            if best_checkpoint:
                mlflow.log_artifact(str(best_checkpoint))
                logger.info(f"Best model saved: {best_checkpoint}")

            # Log final summary
            mlflow.log_metrics({
                "final_train_loss": self.history["train_loss"][-1],
                "final_val_loss": self.history["val_loss"][-1],
                "best_val_loss": best_val_loss,
                "total_epochs": len(self.history["train_loss"]),
            })

        return self.history


def create_model_from_config(
    model_name: str,
    model_config: dict[str, Any],
) -> BaseForecaster:
    """Instantiate a model from configuration.

    Args:
        model_name: Model name (cnn, lstm, gru, nhits, patchtst, tft).
        model_config: Model hyperparameters.

    Returns:
        Instantiated BaseForecaster model.
    """
    register_models()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Remove non-model params
    config = {k: v for k, v in model_config.items() if k not in ("name", "_target_")}
    model = MODEL_REGISTRY[model_name](**config)

    logger.info(f"Created {model}")
    return model


# ─── CLI entrypoint ──────────────────────────────────────────
def main() -> None:
    """CLI entrypoint for training."""
    import sys

    from src.data.loaders.factory import DataLoaderFactory

    # Simple CLI parsing (Hydra integration for production)
    model_name = "lstm"
    dataset_name = "jena"
    horizon = 24
    seed = 42

    for arg in sys.argv[1:]:
        if arg.startswith("model="):
            model_name = arg.split("=")[1]
        elif arg.startswith("dataset="):
            dataset_name = arg.split("=")[1]
        elif arg.startswith("training.forecast_horizon="):
            horizon = int(arg.split("=")[1])
        elif arg.startswith("seed="):
            seed = int(arg.split("=")[1])

    set_seed(seed)

    # Create data loaders
    train_loader, val_loader, test_loader = DataLoaderFactory.create(
        dataset_name=dataset_name,
        forecast_horizon=horizon,
        batch_size=64,
    )

    # Determine input size from data
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[2]

    # Create model
    model = create_model_from_config(model_name, {
        "input_size": input_size,
        "horizon": horizon,
    })

    # Train
    trainer = GenericTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=f"{model_name}_{dataset_name}_{horizon}h",
        seed=seed,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
