# AeroGuard AI

AeroGuard AI is an end-to-end predictive-maintenance system for aircraft engines. It uses NASA C-MAPSS sensor histories to estimate Remaining Useful Life (RUL), explain risk, and generate cited reliability summaries. The project is deliberately built as a production-style system rather than a single notebook.

## Current status

Days 1 through 5 are implemented:

- reproducible download of the official NASA archive;
- strict FD001 schema and business-rule validation;
- training and test RUL-label construction;
- failure-within-30-cycles classification target;
- statistical summaries and four saved visualizations;
- engine-grouped train/validation splitting with zero engine overlap;
- mean, Ridge, Random Forest, and XGBoost regression candidates;
- MAE, RMSE, R², and asymmetric NASA-score evaluation;
- MLflow experiment tracking backed by local SQLite;
- persisted champion model, metadata, predictions, and four model figures;
- 30-cycle sliding-window sequence construction;
- a two-layer PyTorch LSTM with leakage-safe scaling and early stopping;
- fair Random Forest/LSTM comparison on identical sequence endpoints;
- TreeSHAP global and local Random Forest explanations;
- persisted LSTM checkpoint, scaler, predictions, reports, and four Day 3 figures;
- a fingerprinted, heading-chunked reliability knowledge base;
- deterministic local TF-IDF retrieval with Hit@3 and MRR evaluation;
- real Random Forest, LSTM, disagreement, and local SHAP evidence packets;
- a cited offline reliability brief plus optional OpenAI Responses generation;
- protected numerical fields, citation allowlisting, and prompt-injection boundaries;
- a validated FastAPI service over the real model, prediction and retrieval artifacts;
- a responsive operations dashboard with fleet triage, model comparison and engine drill-down;
- server-side filtering, sorting and pagination plus health and request tracing;
- GitHub Actions continuous integration and an opt-in Render deployment blueprint;
- 57 automated tests and five beginner-facing learning guides.

The LSTM is the validation accuracy winner. On the 100 official FD001 test-engine endpoints it achieved **10.34-cycle MAE**, **14.13-cycle RMSE**, and **0.876 R²** using the capped RUL target, improving on the Random Forest's 12.30 MAE and 16.77 RMSE. Random Forest remains the current explainable deployment champion because its asymmetric NASA risk score was better and the implemented SHAP evidence directly explains its predictions. LSTM promotion is held for late-warning risk review.

## Business problem

Unexpected equipment failure is expensive and potentially dangerous. AeroGuard turns multivariate sensor histories into two decision-support outputs:

1. **RUL regression:** approximately how many operating cycles remain before failure?
2. **Near-term risk classification:** is failure expected within the next 30 cycles?

The numerical prediction comes from trained ML models. The RAG/generation layer explains verified model evidence; it cannot invent or replace the prediction.

## Dataset

Source: [NASA C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

Day 1 uses FD001: 100 training engines, 100 test engines, one operating condition, and one high-pressure-compressor degradation mode. Each row represents one engine at one operating cycle and contains three operating settings plus 21 sensor readings.

The source archive is downloaded at runtime and excluded from Git because it is reproducible and NASA lists no explicit license on the current catalog entry. The generated manifest records the source URL, timestamp, size, and SHA-256 checksum.

## Windows quick start

```powershell
cd "path\to\aeroguard-ai"
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run_day1.ps1
powershell -ExecutionPolicy Bypass -File .\run_day2.ps1
powershell -ExecutionPolicy Bypass -File .\run_day3.ps1
powershell -ExecutionPolicy Bypass -File .\run_day4.ps1
powershell -ExecutionPolicy Bypass -File .\test.ps1
powershell -ExecutionPolicy Bypass -File .\run_day5.ps1
```

## macOS/Linux quick start

```bash
make setup
make day1
make day2
make day3
make day4
make day5
make test
make app
```

## Day 1 outputs

Running Day 1 creates:

- `data/raw/CMAPSSData.zip` and extracted FD001 source files;
- `data/raw/source_manifest.json`;
- `data/processed/train_fd001_labeled.csv`;
- `data/processed/test_fd001_labeled.csv`;
- `reports/day1_summary.json`;
- `reports/engine_summary.csv`;
- `reports/sensor_summary.csv`;
- `reports/figures/engine_lifecycle_distribution.png`;
- `reports/figures/rul_distribution.png`;
- `reports/figures/sensor_degradation.png`;
- `reports/figures/sensor_correlation.png`.

## Day 2 outputs

Running Day 2 creates:

- `reports/day2_summary.json` and `reports/day2_metrics.csv`;
- `reports/day2_test_predictions.csv`;
- `reports/day2_feature_importance.csv`;
- four `reports/figures/day2_*.png` evaluation charts;
- `artifacts/models/champion_rul_model.joblib`;
- `artifacts/models/champion_metadata.json`;
- `artifacts/mlruns/mlflow.db` plus tracked run artifacts.

To inspect the experiments on Windows, run `run_mlflow.ps1`, open `http://127.0.0.1:5000`, and press Ctrl+C when finished. On macOS/Linux, use `make mlflow`.

## Day 3 outputs

Running Day 3 creates:

- `reports/day3_summary.json`, `reports/day3_metrics.csv`, and epoch training history;
- `reports/day3_test_predictions.csv` for both Random Forest and LSTM;
- global and local SHAP CSV evidence;
- four `reports/figures/day3_*.png` learning, comparison, and explanation charts;
- `artifacts/models/lstm_rul_model.pt` and `lstm_feature_scaler.joblib`;
- `artifacts/models/lstm_metadata.json` and the explicit deployment decision;
- a tracked `day3_lstm_and_shap` MLflow run.

## Day 4 outputs

Running Day 4 creates:

- `artifacts/rag/knowledge_index.joblib` and a fingerprinted knowledge manifest;
- `reports/day4_retrieval_evaluation.csv` with labeled ranking evidence;
- `reports/day4_reliability_brief.json` with model evidence, retrieved text, and citations;
- `reports/day4_summary.json` with retrieval metrics and safety-boundary metadata.

Day 4 works offline by default. To use optional OpenAI wording, copy `.env.example` to `.env`, set `AEROGUARD_GENERATOR=openai`, add your API key, and select the model. The `.env` file is ignored by Git. The implementation uses Structured Outputs through the [official OpenAI Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create); deterministic post-processing still protects the authoritative RUL and citation allowlist.

## Day 5 application

Day 5 creates `reports/day5_summary.json`, verifies that all earlier artifacts are ready, and serves:

- a real fleet overview and four risk-band counts;
- Random Forest and LSTM evaluation metrics;
- a searchable, sortable, server-paginated engine queue;
- engine-level RUL, model disagreement, TreeSHAP drivers and grounded citations;
- retrieval Hit@3, MRR and individual evaluation cases;
- a health endpoint and automatic OpenAPI documentation.

On Windows, `run_day5.ps1` safely handles a stale AeroGuard process, starts the server, waits for a healthy response and opens the dashboard. Open `http://127.0.0.1:8000`; API docs are at `http://127.0.0.1:8000/docs`. Press `Ctrl+C` to stop it. On macOS/Linux, use `make app`.

`render.yaml` is an optional cloud blueprint, not a claim that the project is deployed. It defaults to the offline local generator and needs an appropriate Render account and compute plan before use.

## Repository structure

```text
aeroguard-ai/
|-- data/                 # Raw, intermediate and processed data
|-- knowledge/            # Citable demonstration reliability guidance
|-- docs/                 # Architecture, data dictionary and learning guide
|-- reports/              # Machine-readable analysis and figures
|-- src/aeroguard/
|   |-- data/             # Download, ingestion, validation and RUL targets
|   |-- analysis/         # Day 1 statistics and figures
|   |-- ml/               # Group splitting, features, metrics and training
|   |-- rag/              # Chunking, retrieval, evaluation, evidence, generation
|   |-- api/              # FastAPI routes, services and operations dashboard
|   `-- cli.py            # Reproducible command-line orchestration
|-- tests/                # Focused automated tests
|-- setup.ps1             # Windows setup
|-- run_day1.ps1          # Windows Day 1 runner
|-- run_day2.ps1          # Windows Day 2 training runner
|-- run_day3.ps1          # Windows Day 3 sequence/explanation runner
|-- run_day4.ps1          # Windows Day 4 grounded-RAG runner
|-- run_day5.ps1          # Windows Day 5 application runner
|-- run_mlflow.ps1        # Windows experiment viewer
|-- render.yaml           # Optional cloud deployment blueprint
`-- pyproject.toml        # Package and dependency definition
```

## Engineering decisions

- **Scripts instead of notebook-only logic:** every important transformation is reusable and testable.
- **Engine-level identity:** `(engine_id, cycle)` is the natural key for one sensor snapshot.
- **No random row splitting:** model partitions are grouped by engine to prevent leakage.
- **Separate train/test RUL rules:** training sequences end at failure; test sequences stop earlier and require the provided truth vector.
- **Honest model selection:** validation RMSE selects the champion before one final official-test evaluation.
- **Operational target:** RUL is capped at 125 during modeling because exact early-life RUL is weakly observable and less useful for maintenance decisions.
- **Auditable experiments:** MLflow stores parameters, metrics, run IDs, and model artifacts in a queryable SQLite-backed tracker.
- **Fair sequence comparison:** Random Forest and LSTM are compared on identical engine-cycle endpoints.
- **Separated governance decisions:** validation identifies the accuracy winner; operational risk and explainability control deployment promotion.
- **Explanation honesty:** TreeSHAP evidence is labeled as a Random Forest explanation and is never claimed to explain the LSTM.
- **Generator boundary:** RAG and optional LLM text describe model evidence but cannot replace the numerical prediction.
- **Auditable retrieval:** chunks have stable citation IDs and hashes, and labeled retrieval questions produce Hit@3 and MRR metrics.
- **Safe-by-default execution:** local deterministic generation is the default; external generation requires explicit configuration.
- **Thin serving layer:** FastAPI exposes reusable service logic and never retrains a model during a dashboard request.
- **No dashboard mock data:** every displayed value comes from generated reports and saved artifacts through the API.
- **Observable failures:** missing artifacts return degraded health or a clear 503 rather than fabricated values.
- **Automated change checks:** GitHub Actions runs the Python 3.11 test suite on pushes and pull requests.
- **Reproducible artifacts:** outputs can be deleted and regenerated from the official source.
- **Honest scope:** C-MAPSS is simulated data, so the project is a decision-support demonstration, not an airworthiness-certified system.

Read [docs/day1_learning_guide.md](docs/day1_learning_guide.md), [docs/day2_learning_guide.md](docs/day2_learning_guide.md), [docs/day3_learning_guide.md](docs/day3_learning_guide.md), [docs/day4_learning_guide.md](docs/day4_learning_guide.md), and [docs/day5_learning_guide.md](docs/day5_learning_guide.md) in order.
