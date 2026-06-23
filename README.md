<div align="center">

# ⛈️ AtmosForge

### Multivariate Meteorological Time Series Forecasting
#### A Production-Grade Deep Learning Benchmarking Framework

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?style=flat-square&logo=mlflow)](https://mlflow.org)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-7B1FA2?style=flat-square)](https://optuna.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)](https://github.com/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

---

## Abstract

**AtmosForge** is a reproducible benchmarking framework for multivariate meteorological time-series forecasting, systematically evaluating six deep learning architectures — from LSTM and CNN-1D baselines to state-of-the-art Temporal Fusion Transformers (TFT), N-HiTS, and PatchTST — across three publicly available datasets and four forecast horizons (1h, 6h, 24h, 72h).

Beyond accuracy comparisons, the framework integrates **uncertainty quantification** via conformal prediction, **feature attribution** via SHAP temporal importance analysis, and **statistical model selection** via the Diebold-Mariano test. The entire pipeline — from data ingestion to serving — is instrumented with MLflow and deployable via Docker.

---

## 🎯 Key Contributions

- **Unified Benchmark** — Standardized evaluation of 6 architectures on 3 real-world datasets across 4 forecast horizons, with deterministic (MAE, RMSE, MAPE) and probabilistic metrics (CRPS, Pinball Loss)
- **Rigorous Model Selection** — Diebold-Mariano statistical tests (p < 0.05) for pairwise comparison; no eyeballing results
- **Full MLOps Instrumentation** — End-to-end pipeline: data ingestion → feature engineering → training → HPO via Optuna → evaluation → serving; every run tracked in MLflow
- **Uncertainty Quantification** — Quantile regression and Monte Carlo Dropout for prediction interval generation
- **Interpretability Layer** — SHAP-based temporal feature importance for all model-dataset pairs
- **Production Inference** — Async FastAPI REST endpoint with Pydantic validation, Dockerized and cloud-deployable
- **Reproducibility Guarantee** — Global seed management, locked dependencies, and `make reproduce` single-command re-execution

---

## 📊 Benchmark Results

### Jena Climate Dataset — Temperature Forecasting

> 🔔 *Placeholder values below. Run `make reproduce SEED=42` to generate your results.*

| Model | MAE ↓ | RMSE ↓ | MAPE ↓ | CRPS ↓ | DM vs Best | Train Time |
|-------|--------|---------|---------|---------|------------|------------|
| CNN-1D (Baseline) | `—` | `—` | `—` | `—` | `—` | `—` |
| LSTM | `—` | `—` | `—` | `—` | `—` | `—` |
| GRU | `—` | `—` | `—` | `—` | `—` | `—` |
| N-HiTS | `—` | `—` | `—` | `—` | `—` | `—` |
| PatchTST | `—` | `—` | `—` | `—` | `—` | `—` |
| **TFT** | `—` | `—` | `—` | `—` | — | `—` |

*Full multi-horizon results (1h/6h/24h/72h) in [`results/benchmark.csv`](results/benchmark.csv)*

---

## 🗃️ Datasets

Three publicly available, programmatically downloadable datasets spanning different resolutions and geographic scopes:

### 1. Jena Climate 2009–2016 (Max Planck Institute for Biogeochemistry)
- **Variables:** 14 atmospheric features (air temperature, pressure, humidity, dew point, wind velocity & direction, etc.)
- **Resolution:** 10-minute intervals (~420,000 observations)
- **Usage in Literature:** François Chollet, *Deep Learning with Python* (2018); TensorFlow official tutorial
- **Direct Download** (no registration):
```bash
make fetch-data DATASET=jena
# or manually:
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.gz -O data/raw/jena_climate.csv.gz
```

### 2. Open-Meteo Historical Weather API (Global, incl. Indonesia)
- **Variables:** 40+ variables including 2m temperature, relative humidity, precipitation, wind speed, surface pressure, shortwave radiation, soil temperature
- **Coverage:** Global — any lat/lon coordinate from 1940 to present; includes all BMKG-equivalent Indonesian stations
- **Resolution:** Hourly or daily
- **Access:** Free API — no registration, no API key required
```bash
make fetch-data DATASET=openmeteo LAT=-6.21 LON=106.85 START=2020-01-01 END=2024-12-31
# Fetches Jakarta weather data via openmeteo-requests
```

### 3. ERA5-Land Hourly Reanalysis (ECMWF / Copernicus Climate Data Store)
- **Variables:** 50+ reanalysis variables at 9km spatial resolution
- **Coverage:** Global, 1950 to present
- **Resolution:** Hourly
- **Credibility:** Used in peer-reviewed publications (Nature, AGU Journals, QJRMS)
- **Access:** Free (one-time registration at cds.climate.copernicus.eu)
```bash
# Requires CDS_API_KEY in .env
make fetch-data DATASET=era5 VARIABLE=2m_temperature,total_precipitation YEAR=2022 REGION=indonesia
```

---

## 🧠 Models

| Architecture | Category | Reference | Key Feature |
|---|---|---|---|
| CNN-1D (Dilated TCN) | Baseline | [Bai et al., 2018](https://arxiv.org/abs/1803.01271) | Receptive field via dilation |
| LSTM | Baseline | [Hochreiter & Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735) | Long-range dependencies |
| GRU | Baseline | [Cho et al., 2014](https://arxiv.org/abs/1406.1078) | Efficient gating mechanism |
| N-HiTS | Advanced | [Challu et al., AAAI 2023](https://arxiv.org/abs/2201.12886) | Hierarchical interpolation |
| PatchTST | Advanced | [Nie et al., ICLR 2023](https://arxiv.org/abs/2211.14730) | Patch-based self-attention |
| TFT | Advanced | [Lim et al., IJF 2021](https://arxiv.org/abs/1912.09363) | Interpretable multi-horizon |

All models output **point forecasts** and **quantile estimates** (q10/q50/q90) for uncertainty quantification.

---

## 🏗️ Architecture

```
atmosforge/
├── configs/                    # Hydra experiment configs (composable YAML)
│   ├── model/                  # Per-model hyperparameters
│   ├── dataset/                # Dataset-specific preprocessing config
│   ├── training/               # Optimizer, scheduler, early stopping
│   └── optuna/                 # HPO search spaces per model
│
├── src/
│   ├── data/
│   │   ├── ingestion/          # Download, cache, validate raw data
│   │   ├── preprocessing/      # Normalization, windowing, split
│   │   └── loaders/            # PyTorch Dataset + DataLoader factories
│   │
│   ├── models/
│   │   ├── base.py             # BaseForecaster abstract class
│   │   ├── baselines/          # LSTM, GRU, CNN-1D
│   │   ├── advanced/           # TFT, N-HiTS, PatchTST
│   │   └── ensemble/           # Weighted averaging, stacking
│   │
│   ├── training/
│   │   ├── trainer.py          # GenericTrainer + MLflow hooks
│   │   └── tuner.py            # OptunaHPOTuner (Bayesian search)
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # MAE, RMSE, CRPS, Pinball, DM-test
│   │   ├── benchmark.py        # Auto-generate benchmark table from MLflow
│   │   └── attribution/        # SHAP temporal importance plots
│   │
│   └── serving/
│       ├── api.py              # FastAPI async inference endpoint
│       ├── schemas.py          # Pydantic request/response models
│       └── model_loader.py     # Load artifacts from MLflow registry
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_results_visualization.ipynb
│
├── tests/
│   ├── unit/                   # Pure function tests (metrics, preprocessing)
│   └── integration/            # End-to-end pipeline tests
│
├── scripts/                    # CLI: fetch, train, benchmark, reproduce
├── results/                    # Auto-generated benchmark outputs (git-tracked)
├── docker/
│   ├── Dockerfile.train        # CUDA-capable training image
│   └── Dockerfile.serve        # Minimal inference image
│
├── .github/workflows/ci.yml    # Lint + typecheck + test + Docker build
├── docker-compose.yml
├── Makefile                    # All commands in one place
├── pyproject.toml              # Modern Python packaging
├── CLAUDE.md                   # AI-assisted development context
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended — zero environment setup)
```bash
git clone https://github.com/yourusername/atmosforge.git && cd atmosforge
cp .env.example .env
docker-compose up train     # Trains all models; MLflow UI available at :5000
docker-compose up serve     # Inference API available at :8000/docs
```

### Option 2: Local Python
```bash
pip install -e ".[dev]"
cp .env.example .env

make fetch-data DATASET=jena
make train MODEL=tft DATASET=jena HORIZON=24
make evaluate
mlflow ui                   # http://localhost:5000
```

### Option 3: Full Benchmark (reproduces all results)
```bash
make reproduce SEED=42      # Runs all model × dataset × horizon combinations
# Outputs: results/benchmark.csv + results/benchmark.md
```

---

## 🔬 MLOps Pipeline

```
[Raw Data] ──► [Ingestion + Validation] ──► [Feature Engineering]
                                                      │
                                                      ▼
                                             [Windowed DataLoaders]
                                                      │
                       ┌──────────────────────────────┘
                       ▼
              [HPO via Optuna] ──► [Training + MLflow Tracking]
                                              │
                       ┌──────────────────────┘
                       ▼
          [Evaluation: DM Test + CRPS + SHAP]
                       │
                       ▼
          [Model Registry (MLflow)] ──► [FastAPI Serving]
```

**Experiment Tracking:** All runs logged to MLflow — parameters, metrics per epoch, model checkpoints, SHAP plots  
**HPO:** Bayesian search via Optuna (50 trials/model by default); nested MLflow runs  
**Model Selection:** Automated Diebold-Mariano test at p < 0.05; winner promoted to registry  
**Deployment:** Single command `make serve` → async REST API with batch inference

---

## 🌐 Inference API

After running `make serve` or `docker-compose up serve`:

```
POST  /predict           →  Forecast from input time series
GET   /predict/batch     →  Batch inference
GET   /models            →  List models in registry
GET   /health            →  Health check
GET   /docs              →  Swagger UI (auto-generated)
```

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[...], [...], ...],
    "horizon": 24,
    "model": "tft",
    "return_quantiles": true
  }'
```

---

## 📦 Key Dependencies

| Category | Library | Version |
|---|---|---|
| Core DL | PyTorch | ≥ 2.2 |
| TS Models | pytorch-forecasting | ≥ 1.0 |
| Config | Hydra-core | ≥ 1.3 |
| Tracking | MLflow | ≥ 2.11 |
| HPO | Optuna | ≥ 3.6 |
| Explainability | SHAP | ≥ 0.45 |
| API | FastAPI + Uvicorn | ≥ 0.110 |
| Data | pandas + polars | ≥ 2.0 |
| Data API | openmeteo-requests | ≥ 0.2 |
| ERA5 | cdsapi | ≥ 0.7 |
| Testing | pytest + pytest-cov | ≥ 8.0 |
| Type Check | mypy | ≥ 1.9 |
| Lint | ruff | ≥ 0.4 |

---

## 🔁 Reproducibility

All results are fully reproducible:
```bash
# Fix all random seeds: PyTorch, NumPy, Python random
export SEED=42
make reproduce SEED=$SEED

# Artifacts: results/benchmark_seed42.csv + MLflow run IDs in results/run_ids.txt
```

Random seeds are managed centrally via `src/utils/seed.py` and propagated through all DataLoaders (`worker_init_fn`), model weight initialization, and Optuna samplers.

---

## 📚 References

1. Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
2. Challu, C. et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI 2023. [arXiv:2201.12886](https://arxiv.org/abs/2201.12886)
3. Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
4. Bai, S. et al. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
5. Diebold, F.X. & Mariano, R.S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics. [DOI](https://doi.org/10.1080/07350015.1995.10524599)
6. Hersbach, H. et al. (2020). *The ERA5 Global Reanalysis.* Quarterly Journal of the Royal Meteorological Society. [DOI](https://doi.org/10.1002/qj.3803)

---

## 📄 License

[MIT License](LICENSE) — see LICENSE for details.

---

<div align="center">
<sub>Built for the Open Meteorological AI community · Contributions welcome</sub>
</div>
