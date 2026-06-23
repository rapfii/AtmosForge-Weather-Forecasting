"""DataLoader factory for creating dataset instances from Hydra config.

Provides a unified interface to instantiate any supported dataset
(Jena, Open-Meteo, ERA5) from Hydra configuration.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.data.base import BaseWeatherDataset
from src.data.ingestion.era5 import ERA5Dataset
from src.data.ingestion.jena import JenaDataset
from src.data.ingestion.openmeteo import OpenMeteoDataset
from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.data.factory")

# Registry of supported datasets
DATASET_REGISTRY: dict[str, type[BaseWeatherDataset]] = {
    "jena": JenaDataset,
    "openmeteo": OpenMeteoDataset,
    "era5": ERA5Dataset,
}

SUPPORTED_DATASETS = list(DATASET_REGISTRY.keys())


class DataLoaderFactory:
    """Factory for creating DataLoaders from Hydra configuration.

    Instantiates the correct dataset class based on the config and
    returns train/val/test DataLoaders.

    Example:
        >>> factory = DataLoaderFactory()
        >>> train, val, test = factory.create(
        ...     dataset_name="jena",
        ...     input_window=168,
        ...     forecast_horizon=24,
        ...     batch_size=64,
        ... )
    """

    @staticmethod
    def create(
        dataset_name: str,
        input_window: int = 168,
        forecast_horizon: int = 24,
        batch_size: int = 64,
        num_workers: int = 4,
        data_dir: str = "data",
        **kwargs: Any,
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
        """Create DataLoaders for the specified dataset.

        Args:
            dataset_name: Name of the dataset ('jena', 'openmeteo', 'era5').
            input_window: Input sequence length.
            forecast_horizon: Prediction horizon.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker processes.
            data_dir: Root data directory.
            **kwargs: Additional dataset-specific parameters.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).

        Raises:
            ValueError: If dataset_name is not supported.
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {SUPPORTED_DATASETS}"
            )

        dataset_class = DATASET_REGISTRY[dataset_name]

        logger.info(
            f"Creating {dataset_name} DataLoaders: "
            f"window={input_window}, horizon={forecast_horizon}, "
            f"batch_size={batch_size}"
        )

        dataset = dataset_class(
            data_dir=data_dir,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        return dataset.get_splits()

    @staticmethod
    def from_hydra_config(
        cfg: Any,
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
        """Create DataLoaders from a Hydra config object.

        Args:
            cfg: Hydra/OmegaConf config with dataset parameters.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        dataset_name = cfg.get("name", cfg.get("dataset", "jena"))
        params = {
            "input_window": cfg.get("input_window", 168),
            "forecast_horizon": cfg.get("forecast_horizon", 24),
            "batch_size": cfg.get("batch_size", 64),
            "num_workers": cfg.get("num_workers", 4),
            "data_dir": cfg.get("data_dir", "data"),
        }

        # Pass through any extra parameters (e.g., latitude, longitude for openmeteo)
        extra_keys = set(cfg.keys()) - set(params.keys()) - {"name", "dataset"}
        for key in extra_keys:
            params[key] = cfg[key]

        return DataLoaderFactory.create(dataset_name=dataset_name, **params)
