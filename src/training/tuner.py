"""Optuna HPO tuner with nested MLflow tracking.

Performs Bayesian hyperparameter optimization using Optuna with:
- Search space defined in YAML configs (configs/optuna/*.yaml)
- Each trial logged as a nested MLflow run
- MedianPruner for early stopping of bad trials
- Returns best config as Hydra-compatible override string
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.models.base import BaseForecaster
from src.training.trainer import GenericTrainer, create_model_from_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("atmosforge.training.tuner")


class OptunaHPOTuner:
    """Hyperparameter optimization using Optuna with MLflow tracking.

    Each Optuna trial creates a nested MLflow run, allowing full
    experiment comparison in the MLflow UI.

    Args:
        model_name: Name of the model to tune.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        n_trials: Number of optimization trials (default: 50).
        max_epochs_per_trial: Max epochs per trial (default: 30).
        patience: Early stopping patience per trial (default: 5).
        experiment_name: MLflow experiment name.
        seed: Random seed.
        search_space: Optional search space dictionary override.
        input_size: Number of input features.
        horizon: Forecast horizon.

    Example:
        >>> tuner = OptunaHPOTuner("lstm", train_loader, val_loader)
        >>> best_config = tuner.run()
    """

    def __init__(
        self,
        model_name: str,
        train_loader: Any,
        val_loader: Any,
        n_trials: int = 50,
        max_epochs_per_trial: int = 30,
        patience: int = 5,
        experiment_name: str = "atmosforge-hpo",
        seed: int = 42,
        search_space: dict[str, Any] | None = None,
        input_size: int = 14,
        horizon: int = 24,
    ) -> None:
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_trials = n_trials
        self.max_epochs_per_trial = max_epochs_per_trial
        self.patience = patience
        self.experiment_name = experiment_name
        self.seed = seed
        self.input_size = input_size
        self.horizon = horizon

        # Default search spaces per model
        self.search_space = search_space or self._get_default_search_space()

    def _get_default_search_space(self) -> dict[str, Any]:
        """Get default search space for the model.

        Returns:
            Dictionary defining the Optuna search space.
        """
        base_space = {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        }

        model_spaces: dict[str, dict[str, Any]] = {
            "lstm": {
                "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
                "num_layers": {"type": "int", "low": 1, "high": 4},
                **base_space,
            },
            "gru": {
                "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
                "num_layers": {"type": "int", "low": 1, "high": 4},
                **base_space,
            },
            "cnn": {
                "hidden_size": {"type": "categorical", "choices": [32, 64, 128]},
                "num_layers": {"type": "int", "low": 2, "high": 6},
                "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
                **base_space,
            },
            "nhits": {
                "hidden_size": {"type": "categorical", "choices": [128, 256, 512]},
                "n_blocks": {"type": "int", "low": 1, "high": 5},
                "n_stacks": {"type": "int", "low": 2, "high": 4},
                **base_space,
            },
            "patchtst": {
                "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
                "n_heads": {"type": "categorical", "choices": [4, 8]},
                "n_layers": {"type": "int", "low": 2, "high": 6},
                "patch_len": {"type": "categorical", "choices": [8, 16, 24]},
                **base_space,
            },
            "tft": {
                "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
                "n_heads": {"type": "categorical", "choices": [2, 4, 8]},
                "n_layers": {"type": "int", "low": 1, "high": 3},
                **base_space,
            },
        }

        return model_spaces.get(self.model_name, base_space)

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters for a trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Sampled hyperparameter dictionary.
        """
        params: dict[str, Any] = {}
        for name, config in self.search_space.items():
            param_type = config["type"]
            if param_type == "float":
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"],
                    log=config.get("log", False),
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(name, config["low"], config["high"])
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, config["choices"])
        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object.

        Returns:
            Validation loss (to minimize).
        """
        # Sample hyperparameters
        params = self._sample_params(trial)

        # Separate model params from training params
        training_params = {}
        model_params = {"input_size": self.input_size, "horizon": self.horizon}

        for key, value in params.items():
            if key in ("learning_rate", "batch_size"):
                training_params[key] = value
            else:
                model_params[key] = value

        try:
            # Create model
            model = create_model_from_config(self.model_name, model_params)

            # Create trainer with nested MLflow run
            run_name = f"trial_{trial.number}_{self.model_name}"

            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_params(params)

                trainer = GenericTrainer(
                    model=model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    max_epochs=self.max_epochs_per_trial,
                    patience=self.patience,
                    seed=self.seed + trial.number,
                    run_name=run_name,
                    learning_rate=training_params.get("learning_rate", 1e-3),
                )

                # Override MLflow to avoid nested start_run issues
                history = trainer.fit()

                best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
                mlflow.log_metric("best_val_loss", best_val_loss)

            return best_val_loss

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")

    def run(self) -> dict[str, Any]:
        """Run the full HPO study.

        Returns:
            Best hyperparameter configuration.
        """
        set_seed(self.seed)

        logger.info(
            f"Starting HPO: model={self.model_name}, "
            f"n_trials={self.n_trials}, "
            f"max_epochs/trial={self.max_epochs_per_trial}"
        )

        # Create Optuna study
        study = optuna.create_study(
            study_name=f"atmosforge_{self.model_name}",
            direction="minimize",
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
            ),
        )

        # Run with MLflow parent run
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=f"hpo_{self.model_name}"):
            mlflow.log_params({
                "model": self.model_name,
                "n_trials": self.n_trials,
                "max_epochs_per_trial": self.max_epochs_per_trial,
            })

            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
            )

            # Log best results
            best_params = study.best_params
            best_value = study.best_value

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_loss", best_value)

            logger.info(f"Best trial: val_loss={best_value:.6f}")
            logger.info(f"Best params: {best_params}")

        return best_params

    def get_hydra_overrides(self, best_params: dict[str, Any]) -> str:
        """Convert best params to Hydra override string.

        Args:
            best_params: Best hyperparameter dict from run().

        Returns:
            Hydra CLI override string.
        """
        overrides = []
        for key, value in best_params.items():
            if key in ("learning_rate", "batch_size"):
                overrides.append(f"training.{key}={value}")
            else:
                overrides.append(f"model.{key}={value}")
        return " ".join(overrides)
