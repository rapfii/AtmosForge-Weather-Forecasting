# AtmosForge — Claude Context

> **Tujuan file ini:** Memory permanen untuk Claude Code. Dibaca setiap sesi. Tidak perlu re-explain project dari awal.

---

## 🎯 Project Overview

| Field | Value |
|---|---|
| **Name** | AtmosForge |
| **Purpose** | Production-grade benchmarking framework untuk multivariate meteorological time-series forecasting |
| **Status** | In Development |
| **Domain** | ML / MLOps / Time Series / Deep Learning |
| **Owner** | [Username GitHub kamu] |

**Satu kalimat:** Framework benchmarking 6 deep learning architectures (LSTM, GRU, CNN-1D, N-HiTS, PatchTST, TFT) pada 3 dataset cuaca nyata di 4 forecast horizon, dengan MLOps penuh.

---

## 🛠️ Tech Stack

| Kategori | Library | Versi |
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
| Type Check | mypy | ≥ 1.9 (strict) |
| Lint | ruff | ≥ 0.4 |
| Package Manager | pip (pyproject.toml) |  |
| Container | Docker + Docker Compose | |

---

## 📁 Project Structure

```
atmosforge/
├── configs/                    ← Hydra YAML (BUKAN hardcode di kode)
│   ├── model/{lstm,gru,cnn,tft,nhits,patchtst}.yaml
│   ├── dataset/{jena,openmeteo,era5}.yaml
│   ├── training/default.yaml
│   └── optuna/{model_name}.yaml
│
├── src/
│   ├── data/
│   │   ├── base.py             ← BaseWeatherDataset (abstract)
│   │   ├── ingestion/          ← Download + cache + validate
│   │   ├── preprocessing/      ← Normalize, window, split (CHRONOLOGICAL)
│   │   └── loaders/            ← PyTorch Dataset + DataLoader
│   │
│   ├── models/
│   │   ├── base.py             ← BaseForecaster (abstract)
│   │   ├── baselines/          ← lstm.py, gru.py, cnn.py
│   │   ├── advanced/           ← tft.py, nhits.py, patchtst.py
│   │   └── ensemble/           ← weighted_avg.py, stacking.py
│   │
│   ├── training/
│   │   ├── trainer.py          ← GenericTrainer (model-agnostic)
│   │   ├── tuner.py            ← OptunaHPOTuner
│   │   └── callbacks/          ← EarlyStopping, LRScheduler hooks
│   │
│   ├── evaluation/
│   │   ├── metrics.py          ← MAE, RMSE, CRPS, Pinball, DM-test
│   │   ├── benchmark.py        ← Auto-generate table dari MLflow runs
│   │   └── attribution/        ← SHAP temporal importance plots
│   │
│   ├── serving/
│   │   ├── api.py              ← FastAPI (async)
│   │   ├── schemas.py          ← Pydantic models
│   │   └── model_loader.py     ← Load dari MLflow artifact store
│   │
│   └── utils/
│       ├── seed.py             ← Global seed management
│       └── logger.py           ← Structured logging (JSON)
│
├── notebooks/                  ← EDA + viz ONLY. Training logic DILARANG di sini
├── tests/unit/ + integration/
├── scripts/                    ← CLI entrypoints
├── results/                    ← Auto-generated outputs (commit ini ke git)
├── docker/
├── .github/workflows/ci.yml
├── Makefile
└── pyproject.toml
```

---

## ⚡ Essential Commands

```bash
# ─── Setup ───────────────────────────────────────────────────
pip install -e ".[dev]"
cp .env.example .env

# ─── Data Ingestion ──────────────────────────────────────────
make fetch-data DATASET=jena
make fetch-data DATASET=openmeteo LAT=-6.21 LON=106.85 START=2020-01-01 END=2024-12-31
make fetch-data DATASET=era5 VARIABLE=2m_temperature YEAR=2022   # butuh CDS_API_KEY

# ─── Training ────────────────────────────────────────────────
make train MODEL=lstm DATASET=jena HORIZON=24
make train MODEL=tft DATASET=jena HORIZON=24
make benchmark ALL=true            # Semua kombinasi model × dataset × horizon

# ─── HPO ─────────────────────────────────────────────────────
make tune MODEL=tft DATASET=jena N_TRIALS=50

# ─── Evaluation ──────────────────────────────────────────────
make evaluate                      # DM test + generate benchmark table
mlflow ui                          # View di http://localhost:5000

# ─── Serving ─────────────────────────────────────────────────
make serve                         # FastAPI di localhost:8000/docs
make docker-build
docker-compose up serve

# ─── Quality ─────────────────────────────────────────────────
pytest tests/ --cov=src --cov-report=term-missing
ruff check src/ tests/
mypy src/

# ─── Reproducibility ─────────────────────────────────────────
make reproduce SEED=42             # Full pipeline, fixed seed
```

---

## 📐 Coding Standards

### Python Style
- **Formatter:** Black (`line-length = 88`)
- **Linter:** ruff (replace flake8 + isort)
- **Type Checking:** mypy **strict mode** — NO `Any` types, always annotate
- **Docstrings:** Google style (lihat contoh di `src/models/base.py`)
- **Imports:** isort-sorted via ruff

### Arsitektur Patterns yang WAJIB diikuti

```python
# ✅ BENAR — semua model wajib inherit ini
class LSTMForecaster(BaseForecaster):
    def forward(self, x: torch.Tensor) -> ForecastOutput: ...

# ✅ BENAR — semua dataset wajib inherit ini
class JenaDataset(BaseWeatherDataset):
    def get_splits(self) -> tuple[DataLoader, DataLoader, DataLoader]: ...

# ✅ BENAR — config via Hydra
cfg = hydra.utils.instantiate(config.model)

# ❌ SALAH — hardcode hyperparameter
lstm = LSTM(hidden_size=128, num_layers=2)  # ← JANGAN

# ✅ BENAR — log ke MLflow
with mlflow.start_run(run_name=f"{model_name}_{dataset}_{horizon}h"):
    mlflow.log_params(OmegaConf.to_container(cfg))
    mlflow.log_metric("val_mae", val_mae, step=epoch)
```

### Naming Conventions

| Type | Convention | Contoh |
|---|---|---|
| File Python | snake_case | `temporal_fusion_transformer.py` |
| Class | PascalCase | `TemporalFusionTransformer`, `JenaDataset` |
| Fungsi/Method | snake_case | `compute_diebold_mariano()`, `get_splits()` |
| Variabel | snake_case | `forecast_horizon`, `batch_size` |
| Hydra Config key | snake_case | `learning_rate`, `num_attention_heads` |
| Konstanta | UPPER_SNAKE | `SUPPORTED_DATASETS`, `MAX_HORIZON` |
| MLflow param | sama dengan config key | |

---

## 🏛️ Architecture Decisions (ADR)

| # | Keputusan | Pilihan | Alasan |
|---|---|---|---|
| 1 | Config system | **Hydra** | Composable YAML, experiment sweep built-in |
| 2 | Training framework | **Custom GenericTrainer** | Lebih kontrol vs Lightning; cleaner untuk benchmarking |
| 3 | Experiment tracking | **MLflow** | Self-hostable, no account, artifact store built-in |
| 4 | HPO | **Optuna** | Ringan, native PyTorch integration, nested MLflow runs |
| 5 | API | **FastAPI** | Async, auto-docs Swagger, Pydantic validation |
| 6 | TFT & N-HiTS impl | **pytorch-forecasting** | Battle-tested, tidak perlu reinvent |
| 7 | Linting | **ruff** | Gantikan flake8 + isort + pyupgrade dalam satu tool |

---

## ⚠️ Critical Rules — ALWAYS FOLLOW

```
JANGAN: Taruh training logic di notebook
JANGAN: Hardcode hyperparameter di kode — SELALU via Hydra config
JANGAN: Commit model weights ke git — artifacts disimpan di MLflow
JANGAN: Random split data temporal — SELALU chronological (train→val→test)
JANGAN: Skip type hints — mypy strict harus pass
JANGAN: Edit files di results/ secara manual — itu auto-generated

SELALU: Log setiap training run ke MLflow
SELALU: set_seed() di awal setiap experiment
SELALU: pytest tests/ sebelum commit perubahan training/evaluation
SELALU: Tambah docstring Google-style ke public functions baru
SELALU: Run `ruff check` dan `mypy src/` sebelum PR
```

---

## 🔒 Data & Reproducibility Rules

- Splits **HARUS** chronological: train 70% / val 15% / test 15%
- **TIDAK BOLEH** ada informasi dari future masuk ke past (data leakage)
- Missing values: forward-fill + tambah boolean flag column (`{col}_missing`)
- Normalization: fit **HANYA** pada train split, transform val dan test pakai scaler yang sama
- Download cache: simpan di `data/raw/` dan skip re-download jika file sudah ada
- Semua random seed via `src/utils/seed.py::set_seed(seed: int)`

---

## 🔗 External Services & APIs

| Service | Purpose | URL | Auth |
|---|---|---|---|
| Open-Meteo | Historical weather (free) | https://open-meteo.com/en/docs | Tidak perlu |
| Copernicus CDS | ERA5 reanalysis | https://cds.climate.copernicus.eu | `CDS_API_KEY` di `.env` |
| MLflow (local) | Experiment tracking | localhost:5000 | Tidak perlu (lokal) |
| FastAPI (local) | Inference endpoint | localhost:8000/docs | `API_KEY` di `.env` |

**Environment variables yang dibutuhkan — lihat `.env.example`:**
```
CDS_API_KEY=your_copernicus_key          # Untuk ERA5
MLFLOW_TRACKING_URI=./mlruns             # Bisa juga remote URI
API_KEY=your_inference_api_key           # Untuk FastAPI endpoint
SEED=42                                  # Global reproducibility seed
```

---

## 🧪 Testing Philosophy

| Layer | Direktori | Scope | Target Coverage |
|---|---|---|---|
| Unit | `tests/unit/` | Pure functions: metrics, preprocessing, windowing | **100%** untuk `metrics.py` |
| Integration | `tests/integration/` | Data fetch → forward pass → metric output | ≥ 70% |
| E2E | `tests/e2e/` | Train 1 epoch, evaluate, serve | 1 happy-path test |

**Rules:**
- Mock SEMUA external API calls (Open-Meteo, CDS) di tests
- Gunakan `pytest.fixture` untuk shared DataLoaders (jangan buat ulang per test)
- Setiap PR wajib pass `pytest tests/unit/` minimum

---

## 📝 Git Workflow

| Aspek | Konvensi |
|---|---|
| Branch name | `feat/model-patchtst`, `fix/dm-test-pvalue`, `exp/tft-openmeteo` |
| Commit format | Conventional Commits: `feat:`, `fix:`, `exp:`, `docs:`, `refactor:`, `test:`, `chore:` |
| PR requirement | CI harus green (lint + typecheck + unit tests) |
| Main branch | Squash merge only; no direct push |
| Model weights | Jangan di-commit — gunakan `mlflow.log_artifact()` |

---

## 📚 Key References (Baca sebelum implementasi model)

| Model | Paper | Link |
|---|---|---|
| TFT | *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (Lim et al., IJF 2021) | https://arxiv.org/abs/1912.09363 |
| N-HiTS | *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting* (Challu et al., AAAI 2023) | https://arxiv.org/abs/2201.12886 |
| PatchTST | *A Time Series is Worth 64 Words* (Nie et al., ICLR 2023) | https://arxiv.org/abs/2211.14730 |
| DM Test | *Comparing Predictive Accuracy* (Diebold & Mariano, 1995) | https://doi.org/10.1080/07350015.1995.10524599 |
| ERA5 | *The ERA5 Global Reanalysis* (Hersbach et al., QJRMS 2020) | https://doi.org/10.1002/qj.3803 |
| CRPS | *Strictly Proper Scoring Rules, Prediction, and Estimation* (Gneiting & Raftery, JASA 2007) | https://doi.org/10.1198/016214506000001437 |

---

## 📎 Additional Context Files

```
@configs/model/         ← Hyperparameter defaults per model
@configs/training/      ← Training regime defaults
@results/benchmark.md   ← Latest benchmark results (auto-generated)
@pyproject.toml         ← All dependencies
```
