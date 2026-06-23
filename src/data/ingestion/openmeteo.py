"""Open-Meteo Historical Weather API ingestion.

Downloads historical weather data from the Open-Meteo API — free,
no API key required. Supports any lat/lon worldwide, including
Indonesian stations (default: Jakarta -6.21, 106.85).

API Docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

logger = setup_logger("atmosforge.data.openmeteo")

# Default hourly variables to fetch
DEFAULT_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "cloud_cover",
    "soil_temperature_0_to_7cm",
]

# Variable name mapping for consistency
VARIABLE_RENAME = {
    "temperature_2m": "T_2m (°C)",
    "relative_humidity_2m": "RH (%)",
    "dew_point_2m": "Tdew (°C)",
    "precipitation": "Precip (mm)",
    "surface_pressure": "P_sfc (hPa)",
    "wind_speed_10m": "WS_10m (m/s)",
    "wind_direction_10m": "WD_10m (deg)",
    "shortwave_radiation": "SW_rad (W/m²)",
    "cloud_cover": "Cloud (%)",
    "soil_temperature_0_to_7cm": "T_soil (°C)",
}


class OpenMeteoDataset(BaseWeatherDataset):
    """Open-Meteo Historical Weather API dataset.

    Fetches hourly historical weather data for any location worldwide.
    No API key required. Data available from 1940 to present.

    Args:
        latitude: Location latitude (default: -6.21 for Jakarta).
        longitude: Location longitude (default: 106.85 for Jakarta).
        start_date: Start date string 'YYYY-MM-DD' (default: '2020-01-01').
        end_date: End date string 'YYYY-MM-DD' (default: '2024-12-31').
        hourly_variables: List of variables to fetch.
        data_dir: Root data directory (default: 'data').
        target_column: Target variable (default: 'T_2m (°C)').
        input_window: Input sequence length (default: 168).
        forecast_horizon: Prediction horizon (default: 24).
        batch_size: DataLoader batch size (default: 64).
        num_workers: DataLoader workers (default: 4).
    """

    def __init__(
        self,
        latitude: float = -6.21,
        longitude: float = 106.85,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        hourly_variables: list[str] | None = None,
        data_dir: str | Path = "data",
        target_column: str = "T_2m (°C)",
        input_window: int = 168,
        forecast_horizon: int = 24,
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
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
        self.hourly_variables = hourly_variables or DEFAULT_HOURLY_VARIABLES
        self._normalizer: WeatherNormalizer | None = None

    def _get_cache_filename(self) -> str:
        """Generate unique cache filename based on query parameters.

        Returns:
            Hash-based filename for caching.
        """
        params_str = (
            f"{self.latitude}_{self.longitude}_"
            f"{self.start_date}_{self.end_date}_"
            f"{'_'.join(sorted(self.hourly_variables))}"
        )
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]  # noqa: S324
        return f"openmeteo_{self.latitude}_{self.longitude}_{params_hash}.csv"

    def download(self) -> Path:
        """Fetch data from Open-Meteo Historical Weather API.

        Uses openmeteo-requests library with caching and retry.
        Caches results in data/raw/ to avoid repeated API calls.

        Returns:
            Path to the cached CSV file.
        """
        cache_file = self.raw_dir / self._get_cache_filename()

        # Check cache
        if cache_file.exists():
            logger.info(f"Open-Meteo data cached at {cache_file}")
            return cache_file

        logger.info(
            f"Fetching Open-Meteo data: lat={self.latitude}, lon={self.longitude}, "
            f"{self.start_date} → {self.end_date}"
        )

        try:
            import openmeteo_requests
            import requests_cache
            from retry_requests import retry

            # Setup cached session with retry
            cache_session = requests_cache.CachedSession(
                str(self.raw_dir / ".openmeteo_cache"),
                expire_after=-1,  # Never expire cache
            )
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            om = openmeteo_requests.Client(session=retry_session)

            # API request
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "hourly": self.hourly_variables,
                "timezone": "auto",
            }

            responses = om.weather_api(url, params=params)
            response = responses[0]

            logger.info(
                f"Location: {response.Latitude():.2f}°N {response.Longitude():.2f}°E, "
                f"Elevation: {response.Elevation():.0f}m"
            )

            # Extract hourly data
            hourly = response.Hourly()
            hourly_data: dict[str, list[float]] = {}

            for i, var_name in enumerate(self.hourly_variables):
                values = hourly.Variables(i).ValuesAsNumpy()
                renamed = VARIABLE_RENAME.get(var_name, var_name)
                hourly_data[renamed] = values.tolist()

            # Create DataFrame
            timestamps = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )

            df = pd.DataFrame(hourly_data, index=timestamps)
            df.index.name = "datetime"

            # Save to cache
            df.to_csv(cache_file)
            logger.info(f"Saved {len(df)} rows to {cache_file}")

            # Save metadata
            meta = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "variables": self.hourly_variables,
                "rows": len(df),
                "elevation": float(response.Elevation()),
            }
            meta_path = cache_file.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        except ImportError:
            logger.error(
                "openmeteo-requests not installed. Run: pip install openmeteo-requests requests-cache retry-requests"
            )
            raise

        return cache_file

    def preprocess(self) -> pd.DataFrame:
        """Preprocess Open-Meteo data.

        Steps:
        1. Load cached CSV
        2. Handle missing values (forward-fill + boolean flags)
        3. Ensure consistent datetime index

        Returns:
            Preprocessed DataFrame with DatetimeIndex.
        """
        csv_path = self.download()
        logger.info(f"Loading Open-Meteo data from {csv_path}")

        df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)

        # Handle missing values
        for col in df.columns:
            if df[col].isna().any():
                missing_count = df[col].isna().sum()
                missing_pct = missing_count / len(df) * 100
                logger.info(
                    f"Column '{col}': {missing_count} missing values "
                    f"({missing_pct:.1f}%, forward-filling)"
                )
                df[f"{col}_missing"] = df[col].isna().astype(int)
                df[col] = df[col].ffill()

        # Backfill any leading NaN (from start of series)
        df = df.bfill()

        logger.info(f"Preprocessed shape: {df.shape}")
        return df

    def get_splits(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create chronological train/val/test DataLoaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        df = self.preprocess()

        # Chronological split
        train_df, val_df, test_df = chronological_split(
            df, train_ratio=self.train_ratio, val_ratio=self.val_ratio
        )

        logger.info(
            f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        # Normalize: fit on train ONLY
        self._normalizer = WeatherNormalizer()
        train_norm = self._normalizer.fit_transform(train_df)
        val_norm = self._normalizer.transform(val_df)
        test_norm = self._normalizer.transform(test_df)

        target_idx = list(df.columns).index(self.target_column)

        # Create datasets
        train_ds = SlidingWindowDataset(train_norm, self.input_window, self.forecast_horizon, target_idx)
        val_ds = SlidingWindowDataset(val_norm, self.input_window, self.forecast_horizon, target_idx)
        test_ds = SlidingWindowDataset(test_norm, self.input_window, self.forecast_horizon, target_idx)

        # Create loaders (NEVER shuffle temporal data)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

        return train_loader, val_loader, test_loader


# ─── CLI entrypoint ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo historical data")
    parser.add_argument("--lat", type=float, default=-6.21, help="Latitude")
    parser.add_argument("--lon", type=float, default=106.85, help="Longitude")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    args = parser.parse_args()

    dataset = OpenMeteoDataset(
        latitude=args.lat, longitude=args.lon,
        start_date=args.start, end_date=args.end,
    )
    path = dataset.download()
    df = dataset.preprocess()
    print(f"Shape: {df.shape}, Date range: {df.index.min()} → {df.index.max()}")
