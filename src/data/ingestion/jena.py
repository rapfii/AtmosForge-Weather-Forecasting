"""Jena Climate 2009-2016 dataset ingestion.

Downloads and caches the Jena Climate dataset from TensorFlow storage.
Contains 14 atmospheric features at 10-minute intervals (~420K observations).

Source: Max Planck Institute for Biogeochemistry
URL: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.gz
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.base import BaseWeatherDataset
from src.data.preprocessing.normalizer import WeatherNormalizer
from src.data.preprocessing.splitter import chronological_split
from src.data.preprocessing.windower import SlidingWindowDataset
from src.utils.logger import setup_logger
from src.utils.seed import worker_init_fn

logger = setup_logger("atmosforge.data.jena")

# Dataset constants
JENA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.gz"
JENA_FILENAME = "jena_climate_2009_2016.csv.gz"
JENA_CSV_FILENAME = "jena_climate_2009_2016.csv"
JENA_MD5 = "be5b9022c0b98667de2e24fece508e1b"  # MD5 of the gzipped file


class _DownloadProgressBar(tqdm):  # type: ignore[type-arg]
    """Progress bar for urllib download."""

    def update_to(
        self, blocks: int = 1, block_size: int = 1, total_size: int | None = None
    ) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


class JenaDataset(BaseWeatherDataset):
    """Jena Climate 2009-2016 dataset.

    14 atmospheric features recorded at 10-minute intervals by the Max Planck
    Institute for Biogeochemistry. Commonly used as a benchmark for time series
    forecasting (e.g., François Chollet, Deep Learning with Python, 2018).

    Features:
        - p (mbar): Atmospheric pressure
        - T (degC): Air temperature (DEFAULT TARGET)
        - Tpot (K): Potential temperature
        - Tdew (degC): Dew point temperature
        - rh (%): Relative humidity
        - VPmax (mbar): Saturation vapor pressure
        - VPact (mbar): Actual vapor pressure
        - VPdef (mbar): Vapor pressure deficit
        - sh (g/kg): Specific humidity
        - H2OC (mmol/mol): Water vapor concentration
        - rho (g/m**3): Air density
        - wv (m/s): Wind velocity
        - max. wv (m/s): Maximum wind velocity
        - wd (deg): Wind direction

    Args:
        data_dir: Root data directory (default: 'data').
        target_column: Target variable (default: 'T (degC)').
        input_window: Input sequence length in timesteps (default: 168).
        forecast_horizon: Prediction horizon in timesteps (default: 24).
        resample_freq: Resample frequency (default: '1h' for hourly).
            Set to None to keep original 10-minute resolution.
        batch_size: DataLoader batch size (default: 64).
        num_workers: DataLoader workers (default: 4).
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        target_column: str = "T (degC)",
        input_window: int = 168,
        forecast_horizon: int = 24,
        resample_freq: str | None = "1h",
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            target_column=target_column,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.resample_freq = resample_freq
        self._normalizer: WeatherNormalizer | None = None

    def _verify_checksum(self, filepath: Path) -> bool:
        """Verify file integrity via MD5 checksum.

        Args:
            filepath: Path to the file to verify.

        Returns:
            True if checksum matches, False otherwise.
        """
        md5 = hashlib.md5()  # noqa: S324
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest() == JENA_MD5

    def download(self) -> Path:
        """Download Jena Climate dataset from TensorFlow storage.

        Caches in data/raw/ and skips re-download if file exists with
        correct checksum.

        Returns:
            Path to the decompressed CSV file.
        """
        csv_path = self.raw_dir / JENA_CSV_FILENAME
        gz_path = self.raw_dir / JENA_FILENAME

        # Check if already downloaded and decompressed
        if csv_path.exists():
            logger.info(f"Jena dataset already exists at {csv_path}")
            return csv_path

        # Check if local copy exists in Data/ directory
        local_data = Path("Data") / JENA_CSV_FILENAME
        if local_data.exists():
            logger.info(f"Found local copy at {local_data}, copying to {csv_path}")
            shutil.copy2(local_data, csv_path)
            return csv_path

        # Download with progress bar
        logger.info(f"Downloading Jena Climate dataset from {JENA_URL}")
        with _DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc="Jena Climate"
        ) as pbar:
            urllib.request.urlretrieve(  # noqa: S310
                JENA_URL, gz_path, reporthook=pbar.update_to
            )

        # Verify checksum
        if not self._verify_checksum(gz_path):
            logger.warning("Checksum mismatch — file may be corrupted")

        # Decompress
        logger.info("Decompressing...")
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        logger.info(f"Jena dataset saved to {csv_path}")
        return csv_path

    def preprocess(self) -> pd.DataFrame:
        """Preprocess Jena Climate data.

        Steps:
        1. Parse Date Time column as DatetimeIndex
        2. Handle missing values via forward-fill + boolean flags
        3. Remove anomalous wind velocity values (wv = -9999)
        4. Optionally resample to hourly frequency

        Returns:
            Preprocessed DataFrame with DatetimeIndex.
        """
        csv_path = self.download()
        logger.info(f"Loading Jena data from {csv_path}")

        df = pd.read_csv(csv_path)

        # Parse datetime
        df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
        df = df.set_index("Date Time")

        # Handle anomalous wind velocity values (-9999.0)
        for col in ["wv (m/s)", "max. wv (m/s)"]:
            if col in df.columns:
                anomalous_mask = df[col] < 0
                if anomalous_mask.any():
                    logger.info(
                        f"Replacing {anomalous_mask.sum()} anomalous values in '{col}'"
                    )
                    df.loc[anomalous_mask, col] = 0.0

        # Handle missing values: forward-fill + boolean flag columns
        for col in df.columns:
            if df[col].isna().any():
                missing_count = df[col].isna().sum()
                logger.info(f"Column '{col}': {missing_count} missing values (forward-filling)")
                df[f"{col}_missing"] = df[col].isna().astype(int)
                df[col] = df[col].ffill()

        # Resample to hourly if requested (original is 10-minute)
        if self.resample_freq is not None:
            logger.info(f"Resampling from 10-min to {self.resample_freq}")
            df = df.resample(self.resample_freq).mean()
            df = df.ffill()  # Fill any NaN from resampling

        logger.info(f"Preprocessed shape: {df.shape}")
        return df

    def get_splits(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create chronological train/val/test DataLoaders.

        CRITICAL: Splits are strictly chronological — no random shuffling.
        StandardScaler is fit ONLY on the training split.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        df = self.preprocess()

        # Chronological split: 70/15/15
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )

        logger.info(
            f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        # Normalize: fit on train ONLY
        self._normalizer = WeatherNormalizer()
        train_normalized = self._normalizer.fit_transform(train_df)
        val_normalized = self._normalizer.transform(val_df)
        test_normalized = self._normalizer.transform(test_df)

        # Get target column index for windowed dataset
        target_idx = list(df.columns).index(self.target_column)

        # Create sliding window datasets
        train_dataset = SlidingWindowDataset(
            data=train_normalized,
            input_window=self.input_window,
            forecast_horizon=self.forecast_horizon,
            target_idx=target_idx,
        )
        val_dataset = SlidingWindowDataset(
            data=val_normalized,
            input_window=self.input_window,
            forecast_horizon=self.forecast_horizon,
            target_idx=target_idx,
        )
        test_dataset = SlidingWindowDataset(
            data=test_normalized,
            input_window=self.input_window,
            forecast_horizon=self.forecast_horizon,
            target_idx=target_idx,
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # NEVER shuffle temporal data
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        return train_loader, val_loader, test_loader


# ─── CLI entrypoint ──────────────────────────────────────────
if __name__ == "__main__":
    dataset = JenaDataset()
    path = dataset.download()
    print(f"Downloaded to: {path}")
    df = dataset.preprocess()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
