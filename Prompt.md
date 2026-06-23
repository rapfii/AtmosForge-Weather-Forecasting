# 🚀 AtmosForge — Master Upgrade Prompt Kit
## Panduan Eksekusi: Dari Notebook Basic → Production ML Framework
### Dibuat dengan framework R.A.C.T.F. dari AI-PROMPT-MASTERY.md

---

> **Cara pakai:** Buka Claude Code → `/init` → jalankan prompt-prompt di bawah **secara berurutan**.
> Setiap prompt adalah satu sprint. Selesaikan satu sebelum lanjut ke berikutnya.
> Referensikan CLAUDE.md setiap kali dengan `@CLAUDE.md`.

---

## 📋 Execution Checklist

```
□ PROMPT 1 — Project Scaffold & Base Classes
□ PROMPT 2 — Data Ingestion & Pipeline
□ PROMPT 3 — Model Implementations
□ PROMPT 4 — Training Infrastructure + MLflow + Optuna
□ PROMPT 5 — Evaluation: Metrics, DM Test, SHAP
□ PROMPT 6 — FastAPI Serving + Docker + CI/CD
□ PROMPT 7 — README Population & Final Polish
```

---

## ═══════════════════════════════════════════
## PROMPT 1 — Project Scaffold & Architecture Foundation
## ═══════════════════════════════════════════

> **Tempel di Claude Code. Jalankan PERTAMA.**

---

```
[ROLE] Kamu adalah Senior ML Engineer dengan 8+ tahun pengalaman membangun production-grade machine learning systems — khususnya di time series forecasting, MLOps, dan Python software architecture.

[CONTEXT]
<context>
Aku sedang membangun "AtmosForge" — sebuah production-grade benchmarking framework untuk multivariate meteorological time series forecasting.

Project ini menggantikan proyek basic (satu Jupyter notebook berisi CNN/GRU/LSTM) dengan arsitektur engineering yang proper, cocok untuk portofolio GitHub level senior AI engineer.

Referensi lengkap ada di @CLAUDE.md — baca dulu sebelum mulai.

Tech stack: PyTorch 2.2+, pytorch-forecasting, Hydra, MLflow, Optuna, SHAP, FastAPI, pytest, Docker.
</context>

[ACTION] Scaffold seluruh struktur proyek AtmosForge. Mulai dari skeleton architecture, bukan implementasi logika. Buat semua file init, abstract base classes, pyproject.toml, Makefile, dan CI config awal.

[TARGET]
<output_format>
Buat file-file berikut dalam urutan ini:

1. pyproject.toml
   → Semua dependencies dari @CLAUDE.md, dengan version pinning
   → Dev extras: pytest, ruff, mypy, black
   → Build system: hatchling atau setuptools

2. src/models/base.py
   → Abstract class BaseForecaster
   → Abstract methods: forward(), configure_optimizers(), training_step()
   → ForecastOutput dataclass: point_forecast, quantiles (q10/q50/q90), metadata
   → Type hints WAJIB, Google-style docstring WAJIB

3. src/data/base.py
   → Abstract class BaseWeatherDataset
   → Abstract methods: download(), preprocess(), get_splits()
   → Type hints WAJIB

4. src/utils/seed.py
   → Fungsi set_seed(seed: int) → None
   → Fix: PyTorch, NumPy, Python random, CUDA, DataLoader workers
   → Tambahkan comment untuk setiap komponen yang di-seed

5. Makefile
   → Semua commands dari @CLAUDE.md::Essential Commands section
   → Tambahkan .PHONY declarations

6. .github/workflows/ci.yml
   → Trigger: push to PR dan main
   → Steps: ruff check → mypy → pytest tests/unit/ → Docker build (tanpa push)
   → Cache: pip packages dan Docker layers
   → Target: selesai < 5 menit

7. .env.example
   → Semua env vars dari @CLAUDE.md

8. docker-compose.yml
   → Services: train, serve, mlflow (UI di :5000)
   → Volumes untuk data/ dan mlruns/

9. Semua __init__.py yang dibutuhkan
</output_format>

[FILTER]
- JANGAN implement business logic dulu — hanya skeleton dan abstract classes
- JANGAN skip type hints atau docstring
- JANGAN hardcode paths — gunakan pathlib.Path
- Kalau ada ambiguity tentang interface, tanya sebelum implement
- PRIORITAS: arsitektur yang clean dan extensible, bukan feature-completeness
```

---

## ═══════════════════════════════════════════
## PROMPT 2 — Data Ingestion & Pipeline
## ═══════════════════════════════════════════

> **Jalankan setelah Prompt 1 selesai dan di-review.**

---

```
[ROLE] Kamu adalah Senior ML Data Engineer dengan spesialisasi time series data pipelines dan ETL untuk meteorological datasets.

[CONTEXT]
<context>
Project scaffold sudah ready. Sekarang kita bangun data layer AtmosForge.

Tiga dataset yang harus didukung (dari @CLAUDE.md):

DATASET 1 — Jena Climate 2009–2016
- Download: wget dari TensorFlow storage (URL ada di README.md)
- Format: CSV, 14 kolom, 10-minute interval
- Target: air temperature (T (degC))

DATASET 2 — Open-Meteo Historical API
- Library: openmeteo-requests
- Parameters: lat, lon, start_date, end_date, hourly variables
- Tidak perlu API key
- Harus support lokasi Indonesia (default: Jakarta -6.21, 106.85)

DATASET 3 — ERA5-Land (via cdsapi)
- Credentials: CDS_API_KEY di environment
- Variables: configurable via Hydra config
- Handle: response lambat (async download dengan progress bar)

Semua dataset harus menghasilkan PyTorch DataLoader dengan format:
- Input window: configurable via Hydra (default: 168 steps)
- Forecast horizon: configurable [1, 6, 24, 72]
- Train/Val/Test split: 70/15/15 CHRONOLOGICAL — TIDAK BOLEH random shuffle
- Normalization: StandardScaler fit pada TRAIN ONLY
- Missing values: forward-fill + tambah boolean flag column
</context>

[ACTION] Implement modul src/data/ lengkap — ingestion, preprocessing, loaders.

[TARGET]
<output_format>
Implement dalam urutan:

1. src/data/ingestion/jena.py — JenaIngestion class
2. src/data/ingestion/openmeteo.py — OpenMeteoIngestion class
3. src/data/ingestion/era5.py — ERA5Ingestion class
4. src/data/preprocessing/normalizer.py — WeatherNormalizer (wrapper StandardScaler)
5. src/data/preprocessing/windower.py — SlidingWindowDataset (PyTorch Dataset)
6. src/data/preprocessing/splitter.py — chronological_split() fungsi
7. src/data/loaders/factory.py — DataLoaderFactory (instantiate dari Hydra config)
8. configs/dataset/jena.yaml
9. configs/dataset/openmeteo.yaml
10. configs/dataset/era5.yaml

Untuk setiap class: sertakan unit test sederhana (berapa baris pun) di akhir blok.
</output_format>

[FILTER]
- JANGAN shuffle temporal data — ever
- JANGAN fit scaler pada val/test — hanya pada train
- HARUS cache hasil download di data/raw/ (cek hash atau file exist)
- HARUS ada progress bar (tqdm) untuk download ERA5
- PRIORITAS: data correctness > performance; kita bisa optimize later
- Kalau ada edge case missing values yang tricky, flag sebagai TODO + comment
```

---

## ═══════════════════════════════════════════
## PROMPT 3 — Model Implementations
## ═══════════════════════════════════════════

> **Jalankan setelah data pipeline bisa fetch data dan bikin DataLoader.**

---

```
[ROLE] Kamu adalah Senior Deep Learning Research Engineer dengan expertise di time series architectures. Kamu tahu paper-paper LSTM, GRU, TCN, TFT, N-HiTS, dan PatchTST secara mendalam.

[CONTEXT]
<context>
Data pipeline sudah ready dan tested. Sekarang implement 6 model.

Semua model HARUS:
1. Inherit dari BaseForecaster di src/models/base.py
2. Punya forward(x: Tensor) → ForecastOutput (point + quantiles)
3. Bisa di-instantiate HANYA dari Hydra config dict (tidak ada hardcode)
4. Punya docstring dengan link ke paper dan estimasi parameter count

BASELINES (implement dari scratch):
→ CNN-1D: Dilated Temporal CNN (TCN-style), receptive field via dilation
→ LSTM: Standard LSTM + residual connection + variational dropout
→ GRU: Standard GRU + residual connection + variational dropout

ADVANCED (wrap pytorch-forecasting, tambahkan adapter):
→ N-HiTS: via pytorch_forecasting.NHiTS
→ PatchTST: cari dari huggingface atau implementasi paper Nie et al. 2023
→ TFT: via pytorch_forecasting.TemporalFusionTransformer

Untuk quantile output:
- Baselines: tambahkan quantile head terpisah (3 output: q10/q50/q90)
- TFT/N-HiTS sudah support quantile natively di pytorch-forecasting
</context>

[ACTION] Implement semua 6 model, satu per satu, dari paling simple ke kompleks.

[TARGET]
<output_format>
Per model, buat:
1. src/models/[baselines atau advanced]/[nama].py — full implementation
2. configs/model/[nama].yaml — default hyperparameters
3. Forward pass test (5-10 baris) menggunakan dummy tensor

Urutan: CNN-1D → LSTM → GRU → N-HiTS → PatchTST → TFT

Format docstring per model:
"""
[Nama Model]

Reference:
    [Nama author et al. (tahun). Judul paper. Venue. URL]

Architecture:
    [Deskripsi singkat 2-3 kalimat]

Args:
    input_size: ...
    hidden_size: ...
    [dst]

Approximate parameters:
    ~[X]M parameters dengan config default
"""
</output_format>

[FILTER]
- JANGAN reinvent TFT dan N-HiTS dari scratch — pakai pytorch-forecasting
- JANGAN buat model terlalu complex — maintainability > bleeding-edge SOTA
- Interface HARUS konsisten: semua model terima (batch, seq_len, n_features) → ForecastOutput
- Kalau pytorch-forecasting punya interface berbeda, buat adapter class
- HARUS ada parameter count di docstring (estimasi kasar OK)
```

---

## ═══════════════════════════════════════════
## PROMPT 4 — Training Infrastructure + MLflow + Optuna
## ═══════════════════════════════════════════

> **Jalankan setelah minimal LSTM bisa forward pass dengan benar.**

---

```
[ROLE] Kamu adalah MLOps Engineer berpengalaman dalam membangun production training pipelines — experiment tracking, reproducibility, hyperparameter optimization.

[CONTEXT]
<context>
Models sudah ada. Sekarang bangun training infrastructure.

GenericTrainer harus:
- Agnostik terhadap model (tidak ada if/else per model)
- Log ke MLflow: params, metrics per epoch, best checkpoint artifact
- EarlyStopping: patience=10, monitor val_loss
- Gradient clipping: default max_norm=1.0
- Mixed precision: torch.cuda.amp jika CUDA available
- Checkpoint save/load: support resume dari checkpoint
- Semua config via Hydra, tidak ada hardcode

OptunaHPOTuner harus:
- Search space didefinisikan di configs/optuna/{model}.yaml
- Setiap trial di-log sebagai nested MLflow run
- Return best config sebagai Hydra override string
- Support pruning (Optuna MedianPruner)
- Default: 50 trials per model

LR Schedulers yang harus didukung (configurable via Hydra):
- ReduceLROnPlateau
- CosineAnnealingLR
- OneCycleLR
</context>

[ACTION] Implement src/training/trainer.py, src/training/tuner.py, dan semua configs/training/.

[TARGET]
<output_format>
1. src/training/trainer.py — GenericTrainer class (full implementation)
2. src/training/tuner.py — OptunaHPOTuner class (full implementation)
3. src/training/callbacks/early_stopping.py
4. configs/training/default.yaml
5. configs/optuna/lstm.yaml (sebagai template, sisanya bisa ikut format ini)

Integration test yang harus bisa jalan setelah ini:
→ Train LSTM 3 epochs pada Jena tiny subset (100 rows)
→ Assert: MLflow run terbuat, val_loss ter-log, checkpoint tersimpan
</output_format>

[FILTER]
- HARUS ada checkpoint save dan load (resume training)
- HARUS set_seed() dipanggil di awal setiap training run
- HARUS log model artifact ke MLflow (bukan hanya metrics)
- JANGAN ada state mutable di luar experiment directory
- PRIORITAS: reproducibility dan correctness > training speed
- Kalau Optuna dan MLflow ada conflict di nested run setup, tanya dulu
```

---

## ═══════════════════════════════════════════
## PROMPT 5 — Evaluation: Metrics, DM Test, SHAP, Benchmark Table
## ═══════════════════════════════════════════

> **Jalankan setelah bisa train minimal 2 model sampai selesai.**

---

```
[ROLE] Kamu adalah ML Research Engineer dengan background kuat di statistik dan model interpretability. Kamu familiar dengan proper scoring rules dan statistical significance testing.

[CONTEXT]
<context>
Training pipeline sudah jalan. Sekarang kita bangun evaluation layer yang rigorous — ini yang membedakan project portfolio biasa dengan yang kelihatan seperti dikerjakan researcher.

Metrics yang harus ada:

DETERMINISTIC:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error) — handle zero-division
- sMAPE (Symmetric MAPE)

PROBABILISTIC (untuk quantile outputs):
- CRPS (Continuous Ranked Probability Score) — proper scoring rule, hitung per step lalu average
- Pinball Loss (Quantile Loss) — untuk q10, q50, q90 secara terpisah
- Coverage (apakah 80% aktual jatuh dalam interval [q10, q90])

STATISTICAL COMPARISON:
- Diebold-Mariano Test — two-tailed, pakai scipy.stats
- Hasil: test statistic, p-value, apakah signifikan di p < 0.05
- Harus bisa compare any pair of models

SHAP:
- DeepExplainer untuk LSTM/GRU/CNN
- KernelExplainer untuk model lain (lebih lambat, OK)
- Output: temporal importance plot — welche features paling berpengaruh di timestep mana
- Save ke results/shap/{model}_{dataset}_{horizon}h.png
- Tambahkan tqdm progress bar (bisa lambat)

BENCHMARK TABLE:
- Auto-generate dari MLflow runs setelah semua training selesai
- Format: markdown + CSV
- Asterisk (*) untuk nilai terbaik per kolom
- Save ke results/benchmark.md dan results/benchmark.csv
</context>

[ACTION] Implement src/evaluation/ lengkap + scripts/run_benchmark.py

[TARGET]
<output_format>
1. src/evaluation/metrics.py — semua metrics + DM test
2. src/evaluation/attribution/shap_explainer.py
3. src/evaluation/benchmark.py — pull dari MLflow, generate table
4. scripts/run_benchmark.py — CLI entrypoint untuk full benchmark run
5. tests/unit/test_metrics.py — WAJIB ada test untuk:
   - MAE dengan nilai diketahui (numerical assert)
   - CRPS formula correctness
   - DM test dengan perfect forecast (harus p > 0.05 vs baseline)
   - Pinball loss monotonicity check
</output_format>

[FILTER]
- CRPS harus proper scoring rule — verifikasi formula dari Gneiting & Raftery (2007)
- DM test harus two-tailed (bukan one-tailed)
- SHAP computation bisa SANGAT lambat — berikan opsi `--n_background` untuk sampling
- Benchmark table harus include: model name, MAE, RMSE, CRPS, DM p-value vs best
- PRIORITAS: metric correctness adalah non-negotiable — unit tests 100% coverage untuk metrics.py
```

---

## ═══════════════════════════════════════════
## PROMPT 6 — FastAPI Serving + Docker + CI/CD
## ═══════════════════════════════════════════

> **Jalankan setelah evaluation pipeline bisa generate benchmark table.**

---

```
[ROLE] Kamu adalah DevOps/MLOps Engineer yang tahu cara membungkus ML models menjadi production service — containerization, async serving, dan CI/CD pipelines.

[CONTEXT]
<context>
ML pipeline sudah lengkap. Sekarang kita tambahkan production serving layer.

FastAPI endpoint requirements:
- POST /predict — terima time series input, return forecast + confidence intervals
- GET /predict/batch — batch inference (list of inputs)
- GET /models — list available models dari MLflow registry
- GET /health — health check (return status + model yang di-load)
- Auth: simple API key via header X-API-Key (baca dari env var)
- Auto-docs via Swagger UI (/docs)
- ASYNC: semua inference handler harus async

Pydantic schemas (src/serving/schemas.py):
- ForecastRequest: features (list), horizon (int), model (str), return_quantiles (bool)
- ForecastResponse: forecast, quantiles (optional), model_used, inference_time_ms
- HealthResponse: status, loaded_model, mlflow_uri, version

Docker:
- Dockerfile.train: base image python:3.10-slim + CUDA jika CUDA_VERSION arg ada
- Dockerfile.serve: minimal image, hanya inference dependencies, target < 2GB
- docker-compose.yml: services train, serve, mlflow dengan proper volumes

GitHub Actions CI (update dari Prompt 1):
- Jobs: lint (ruff) → typecheck (mypy) → test (pytest unit) → docker-build
- Semua jobs parallel kecuali docker-build (depends on test)
- Cache pip dengan hash dari pyproject.toml
- Total target: < 5 menit
</context>

[ACTION] Implement src/serving/ + update docker/ + finalize .github/workflows/ci.yml + update Makefile

[TARGET]
<output_format>
1. src/serving/api.py — FastAPI application (async, lengkap)
2. src/serving/schemas.py — Pydantic models
3. src/serving/model_loader.py — Load model dari MLflow registry
4. docker/Dockerfile.train
5. docker/Dockerfile.serve (target < 2GB final image)
6. docker-compose.yml (final version dengan semua services)
7. .github/workflows/ci.yml (finalized)
8. Update Makefile: target serve, docker-build, docker-push

Tambahkan integration test: POST /predict dengan dummy payload → assert 200 OK
</output_format>

[FILTER]
- FastAPI inference HARUS async (async def + await)
- Docker image serve: JANGAN include training dependencies (torch-cuda jika tidak perlu)
- CI JANGAN push Docker image ke registry (hanya build untuk validate)
- JANGAN hardcode API key di kode — baca dari environment
- Kalau model belum ada di registry (pertama kali), /predict harus return 503 + pesan jelas
```

---

## ═══════════════════════════════════════════
## PROMPT 7 — README Population & Final Polish
## ═══════════════════════════════════════════

> **Jalankan TERAKHIR setelah semua experiment sudah berjalan dan benchmark.csv sudah ada.**

---

```
[ROLE] Kamu adalah Technical Writer sekaligus Senior ML Engineer yang tahu apa yang recruiter dan hiring manager di top tech companies perhatikan di GitHub portfolio.

[CONTEXT]
<context>
Seluruh pipeline AtmosForge sudah selesai dan benchmark sudah dijalankan.
File hasil tersedia di: results/benchmark.csv

README.md saat ini ada di @README.md — ada tabel benchmark yang berisi placeholder ("—").

Kita perlu:
1. Isi tabel benchmark dengan data asli dari results/benchmark.csv
2. Tambahkan satu paragraf "Key Findings" yang summarize hasil
3. Pastikan semua badge (CI, Python version, dll) mengarah ke URL yang benar
4. Tambahkan GIF demo atau screenshot jika ada (MLflow UI, FastAPI Swagger, SHAP plot)
5. Review semua link — pastikan tidak ada broken link
6. Cek bahwa "Quick Start" commands semua benar dan bisa dijalankan dari fresh clone
</context>

[ACTION] Finalize README.md dengan real benchmark results dan pastikan semua konten akurat.

[TARGET]
<output_format>
1. Updated README.md dengan:
   - Tabel benchmark terisi data nyata dari results/benchmark.csv
   - Paragraf "Key Findings" (3-5 bullet points tentang insight dari hasil)
   - Badge URLs yang sudah di-fix ke repo yang benar
   - Section "Reproducing Results" yang jelas

2. Checklist pre-publish (output sebagai markdown checklist):
   - [ ] Semua commands di Quick Start dicoba dari scratch
   - [ ] Tidak ada hardcoded path atau username
   - [ ] .gitignore mencakup: data/raw/, mlruns/, checkpoints/, .env, __pycache__
   - [ ] results/benchmark.csv dan results/benchmark.md ada dan up-to-date
   - [ ] LICENSE file ada
   - [ ] requirements.txt atau pyproject.toml up-to-date
</output_format>

[FILTER]
- JANGAN exaggerate hasil — tampilkan angka apa adanya
- Kalau TFT tidak significantly better dari GRU (DM test p > 0.05), katakan itu di Key Findings
- Tone: technical dan objektif, bukan marketing
- PRIORITAS: accuracy > presentation (recruiter yang baik membaca angka, bukan hype)
```

---

## 💡 Tips Eksekusi

```
SEBELUM MULAI:
□ /init di root project → Claude akan scan dan buat initial CLAUDE.md draft
□ Edit CLAUDE.md dengan konten dari file CLAUDE.md yang sudah kita buat
□ Test: claude "baca @CLAUDE.md dan ringkas project ini dalam 3 kalimat"

SAAT MENJALANKAN TIAP PROMPT:
□ Jalankan satu prompt → review output → commit → baru lanjut
□ Gunakan /plan sebelum prompt yang besar (Prompt 2, 3, 5)
□ Gunakan /diff setelah setiap prompt untuk review semua perubahan
□ Kalau context mulai penuh, gunakan /compact

QUALITY GATES (jangan lanjut kalau belum pass):
□ Prompt 1 selesai → pyproject.toml bisa: pip install -e ".[dev]"
□ Prompt 2 selesai → make fetch-data DATASET=jena bisa jalan
□ Prompt 3 selesai → pytest tests/unit/test_models.py PASS
□ Prompt 4 selesai → satu MLflow run tercatat di mlruns/
□ Prompt 5 selesai → results/benchmark.csv ada (meskipun 1 model saja dulu)
□ Prompt 6 selesai → curl localhost:8000/health → 200 OK

JIKA STUCK:
→ "Explain apa yang dilakukan [fungsi] ini dan kenapa" (jangan langsung minta fix)
→ "/rewind ke [checkpoint]" jika arah melenceng
→ Paste error + "Diagnosa penyebab root cause sebelum suggest fix"
```

---

## 🎯 Target GitHub Portofolio Akhir

Ketika semua selesai, repo ini harus menunjukkan:

```
✓ Kemampuan arsitektur software (modular, testable, configurable)
✓ MLOps maturity (experiment tracking, HPO, reproducibility)
✓ Research rigor (DM test, proper scoring rules, references ke papers)
✓ Production readiness (Docker, FastAPI, CI/CD)
✓ Code quality (type hints, mypy, ruff, test coverage ≥ 80%)
✓ Documentation (README kelas world-class, CLAUDE.md, docstrings)
✓ Data engineering (3 dataset sources, proper pipeline)
```

Ini bukan lagi "proyek mahasiswa" — ini **research engineering portfolio** yang bisa dibawa ke interview di perusahaan riset AI maupun industri.
