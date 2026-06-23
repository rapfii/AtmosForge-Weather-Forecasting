"""CLI entrypoint for running full benchmark.

Usage:
    python -m scripts.run_benchmark --all --seed=42
    python -m scripts.run_benchmark --model=lstm --dataset=jena --horizon=24
    python -m scripts.run_benchmark --evaluate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import BenchmarkGenerator
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("atmosforge.benchmark")

# All valid combinations
MODELS = ["cnn", "lstm", "gru", "nhits", "patchtst", "tft"]
DATASETS = ["jena"]  # Start with jena, add openmeteo/era5 as available
HORIZONS = [1, 6, 24, 72]


def run_single(
    model_name: str, dataset_name: str, horizon: int, seed: int
) -> None:
    """Run a single model-dataset-horizon combination.

    Args:
        model_name: Model name.
        dataset_name: Dataset name.
        horizon: Forecast horizon.
        seed: Random seed.
    """
    from src.data.loaders.factory import DataLoaderFactory
    from src.training.trainer import GenericTrainer, create_model_from_config

    set_seed(seed)

    logger.info(
        f"Running: {model_name} × {dataset_name} × {horizon}h (seed={seed})"
    )

    # Create data
    train_loader, val_loader, test_loader = DataLoaderFactory.create(
        dataset_name=dataset_name,
        forecast_horizon=horizon,
        batch_size=64,
    )

    # Get input size from data
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

    logger.info(f"Completed: {model_name} × {dataset_name} × {horizon}h")


def run_full_benchmark(seed: int = 42) -> None:
    """Run all model × dataset × horizon combinations.

    Args:
        seed: Random seed.
    """
    total = len(MODELS) * len(DATASETS) * len(HORIZONS)
    current = 0

    for dataset in DATASETS:
        for horizon in HORIZONS:
            for model in MODELS:
                current += 1
                logger.info(f"[{current}/{total}] {model} × {dataset} × {horizon}h")
                try:
                    run_single(model, dataset, horizon, seed)
                except Exception as e:
                    logger.error(f"Failed: {model} × {dataset} × {horizon}h: {e}")
                    continue


def evaluate_only() -> None:
    """Generate benchmark table from existing MLflow runs."""
    generator = BenchmarkGenerator()
    table = generator.generate()
    csv_path, md_path = generator.save(table)
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(generator.to_markdown(table))
    print(f"\nSaved to: {csv_path}, {md_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AtmosForge Benchmark Runner")
    parser.add_argument("--model", type=str, default="lstm", choices=MODELS)
    parser.add_argument("--dataset", type=str, default="jena", choices=DATASETS)
    parser.add_argument("--horizon", type=int, default=24, choices=HORIZONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all combinations")
    parser.add_argument("--evaluate-only", action="store_true", help="Generate table only")
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_only()
    elif args.all:
        run_full_benchmark(args.seed)
        evaluate_only()
    else:
        run_single(args.model, args.dataset, args.horizon, args.seed)


if __name__ == "__main__":
    main()
