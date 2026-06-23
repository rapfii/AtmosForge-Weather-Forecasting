"""Auto-generate benchmark table from MLflow runs.

Pulls metrics from all completed MLflow runs and generates
a formatted benchmark table in both Markdown and CSV formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.evaluation.benchmark")


class BenchmarkGenerator:
    """Generate benchmark comparison tables from MLflow runs.

    Queries MLflow for completed experiment runs and compiles
    metrics into a formatted comparison table.

    Args:
        experiment_name: MLflow experiment name to query.
        tracking_uri: MLflow tracking URI (default: './mlruns').
        metrics: List of metric names to include.
        results_dir: Directory to save results.

    Example:
        >>> gen = BenchmarkGenerator()
        >>> table = gen.generate()
        >>> gen.save(table)
    """

    def __init__(
        self,
        experiment_name: str = "atmosforge",
        tracking_uri: str = "./mlruns",
        metrics: list[str] | None = None,
        results_dir: str | Path = "results",
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.metrics = metrics or [
            "mae", "rmse", "mape", "crps", "coverage_80",
        ]
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def pull_from_mlflow(self) -> pd.DataFrame:
        """Pull experiment results from MLflow.

        Returns:
            DataFrame with model results.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                logger.warning(
                    f"Experiment '{self.experiment_name}' not found. "
                    "Returning empty DataFrame."
                )
                return pd.DataFrame()

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["metrics.best_val_loss ASC"],
            )

            if runs.empty:
                logger.warning("No finished runs found.")
                return pd.DataFrame()

            logger.info(f"Found {len(runs)} completed runs")
            return runs

        except ImportError:
            logger.error("MLflow not installed.")
            return pd.DataFrame()

    def generate(
        self, runs_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Generate benchmark table.

        Args:
            runs_df: Optional pre-loaded runs DataFrame.

        Returns:
            Formatted benchmark DataFrame.
        """
        if runs_df is None:
            runs_df = self.pull_from_mlflow()

        if runs_df.empty:
            return self._create_placeholder_table()

        # Extract relevant columns
        results = []
        for _, run in runs_df.iterrows():
            row: dict[str, Any] = {
                "Model": run.get("params.model", "Unknown"),
            }

            # Add metrics
            for metric in self.metrics:
                col_name = f"metrics.{metric}"
                if col_name in run.index:
                    row[metric.upper()] = run[col_name]
                else:
                    row[metric.upper()] = None

            # Training time
            if "metrics.epoch_time_s" in run.index and "metrics.total_epochs" in run.index:
                total_time = run.get("metrics.epoch_time_s", 0) * run.get("metrics.total_epochs", 0)
                row["Train Time"] = f"{total_time:.0f}s"
            else:
                row["Train Time"] = "—"

            results.append(row)

        table = pd.DataFrame(results)

        # Mark best values per metric column
        for col in table.columns:
            if col not in ("Model", "Train Time") and table[col].notna().any():
                numeric_vals = pd.to_numeric(table[col], errors="coerce")
                if numeric_vals.notna().any():
                    best_idx = numeric_vals.idxmin()
                    if best_idx is not None:
                        val = table.at[best_idx, col]
                        table.at[best_idx, col] = f"**{val}***"

        return table

    def _create_placeholder_table(self) -> pd.DataFrame:
        """Create placeholder table when no runs are available.

        Returns:
            DataFrame with placeholder values.
        """
        models = ["CNN-1D", "LSTM", "GRU", "N-HiTS", "PatchTST", "TFT"]
        return pd.DataFrame({
            "Model": models,
            **{m.upper(): ["—"] * len(models) for m in self.metrics},
            "Train Time": ["—"] * len(models),
        })

    def to_markdown(self, table: pd.DataFrame) -> str:
        """Convert benchmark table to Markdown format.

        Args:
            table: Benchmark DataFrame.

        Returns:
            Markdown-formatted table string.
        """
        return table.to_markdown(index=False)

    def save(
        self, table: pd.DataFrame, filename_prefix: str = "benchmark"
    ) -> tuple[Path, Path]:
        """Save benchmark table as CSV and Markdown.

        Args:
            table: Benchmark DataFrame.
            filename_prefix: Base filename.

        Returns:
            Tuple of (csv_path, md_path).
        """
        csv_path = self.results_dir / f"{filename_prefix}.csv"
        md_path = self.results_dir / f"{filename_prefix}.md"

        table.to_csv(csv_path, index=False)
        with open(md_path, "w") as f:
            f.write(f"# AtmosForge Benchmark Results\n\n")
            f.write(self.to_markdown(table))
            f.write("\n\n*\\* indicates best value per column*\n")

        logger.info(f"Benchmark saved: {csv_path}, {md_path}")
        return csv_path, md_path
