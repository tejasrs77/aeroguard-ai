# Day 5 learning guide: API, dashboard, testing and deployment

Day 5 turns the work from Days 1-4 into an application another person can use. The models are not retrained when a dashboard page is opened. The app reads the saved predictions, reports, SHAP evidence and RAG index, then presents them through a stable API and browser interface.

## 1. Complete Day 5 flow

```text
Browser dashboard
       | HTTP request, for example GET /api/engines
       v
FastAPI route and Pydantic validation
       v
Reusable service function
       +--> Day 1-4 JSON and CSV reports
       +--> saved Random Forest and LSTM artifacts
       +--> saved TF-IDF knowledge index
       v
Validated JSON response
       v
Metrics, fleet queue, charts and cited engine brief
```

The dashboard has no invented KPI values. JavaScript fetches them from FastAPI, and FastAPI reads the generated project artifacts.

## 2. What an API is

An API is a defined way for programs to communicate. Here, the browser sends an HTTP request to FastAPI. FastAPI returns JSON, a structured text format JavaScript can read.

```text
Request:  GET /api/engines?page=1&page_size=10
Response: {"items": [...], "pagination": {...}}
```

- `GET` asks for data without changing it.
- `POST` sends a body to perform a calculation.
- A route combines an HTTP method and path.
- A query parameter appears after `?`, such as `page=2`.
- A path parameter is inside the URL, such as `34` in `/api/engines/34/brief`.
- A request body is JSON sent with a POST request.

## 3. AeroGuard endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Says whether all required model, report and RAG artifacts exist |
| `GET /api/overview` | Returns dataset, fleet, model and retrieval KPIs |
| `GET /api/models` | Returns comparable Random Forest and LSTM metrics |
| `GET /api/retrieval` | Returns Hit@3, MRR and labeled retrieval results |
| `GET /api/engines` | Returns a searchable, filterable, paginated engine queue |
| `GET /api/engines/{engine_id}/brief` | Returns verified evidence, citations and a brief |
| `POST /api/copilot` | Builds a local or explicitly enabled OpenAI brief for a question |

FastAPI automatically creates interactive documentation at `/docs`. Developers can inspect and try every endpoint there.

## 4. Validation and status codes

Pydantic validates incoming values before application logic runs. An engine ID must be positive, page size is limited to 50, and generator mode must be `local` or `openai`.

- `200 OK`: request worked.
- `400 Bad Request`: valid request shape, but the operation could not complete.
- `404 Not Found`: requested engine does not exist.
- `422 Unprocessable Entity`: FastAPI/Pydantic rejected invalid input.
- `503 Service Unavailable`: a required artifact is missing. The system refuses to return partial or fake data.

Each response includes a request ID and processing time. The ID links a user's problem to a server log, while processing time exposes latency.

## 5. Why route and service layers are separate

`api/main.py` handles routes, validation, status codes, CORS and headers. `api/service.py` handles AeroGuard logic: reading reports, calculating risk counts, filtering engines and producing briefs.

This means service logic can be tested without a server, another interface can reuse it, HTTP changes do not rewrite model logic, and errors have one clear translation into API responses.

## 6. How the fleet queue works

Day 3 has one endpoint prediction per official test engine. Day 5 derives:

```text
risk band = deterministic threshold applied to Random Forest RUL
model disagreement = absolute(Random Forest RUL - LSTM RUL)
```

Random Forest is the deployment source of truth. LSTM is advisory. Large disagreement means a human should inspect the case; it does not mean the application should silently average the models.

Filtering and sorting happen before pagination. Lowest RUL is the default sort, so urgent engines appear first. The benchmark actual RUL exists only because this is an evaluation dataset. A live system would not know future failure time.

## 7. Engine brief and RAG boundary

When an engine is selected:

1. Random Forest produces the authoritative RUL.
2. LSTM supplies an advisory estimate.
3. TreeSHAP explains which Random Forest inputs moved its prediction.
4. TF-IDF retrieves relevant handbook chunks.
5. The generator writes from verified evidence and retrieved text.
6. Code protects numerical fields and allows only retrieved citation IDs.

The generator is a communication layer, not the prediction model. It cannot become the RUL source of truth.

## 8. Browser dashboard

The dashboard provides real overview KPIs, model comparison, risk distribution, search, risk filtering, sorting, server-side pagination, an evidence panel with SHAP drivers and citations, retrieval metrics, skeleton loading states, clear errors and keyboard-accessible rows.

The frontend and backend are served from the same FastAPI process. This simplifies setup and avoids cross-origin problems in this demonstration. CORS remains restricted to local addresses.

## 9. Testing strategy

- **Service tests** use tiny prediction files to verify health, sorting, filtering, pagination and risk bands.
- **API tests** use FastAPI's test client to verify status codes, validation, error translation, headers and dashboard delivery.
- **Real smoke checks** start the assembled app and request health, overview, fleet queue, engine 34 brief, dashboard and API docs against the actual artifacts.

Unit tests are fast and precise. Smoke tests show that assembled components communicate correctly. Neither is evidence that the system is safe for real aircraft.

## 10. CI and cloud blueprint

GitHub Actions runs tests on Python 3.11 for pushes and pull requests. This is continuous integration: every change gets an automatic quality check.

`render.yaml` is a deployment blueprint, not a claim that AeroGuard is already hosted. It installs the code, regenerates required artifacts, starts Uvicorn and checks `/api/health`. It needs a Render account, internet access for the NASA download and enough memory for PyTorch. OpenAI generation stays off by default.

Production training should not occur during web deployment. A scheduled pipeline should publish an approved, versioned model to object storage or a model registry. The web service should load that approved version independently.

## 11. What production would add

- Stream sensors through Kafka or a cloud message service.
- Store features in a warehouse or feature store instead of CSV files.
- Schedule training and validation with Airflow, Prefect or a managed pipeline.
- Register and approve model versions before deployment.
- Add login, role-based access, rate limits and audit logs.
- Cache repeated briefs and put expensive generation on a job queue.
- Monitor input drift, prediction drift, latency, errors and maintenance outcomes.
- Calibrate thresholds with engineers and false-negative costs.
- Use approved documents with access and version governance.
- Use shadow and canary releases before changing the champion.

## 12. Interview questions and short answers

**What did Day 5 add?**  
I exposed real model and RAG artifacts through a validated FastAPI service, built a responsive dashboard, and added API tests, CI and a cloud blueprint.

**Why FastAPI?**  
It is lightweight, validates Python types with Pydantic, generates OpenAPI docs and integrates directly with Python ML code.

**Does the dashboard use mock data?**  
No. Every number comes through the API from generated Day 1-4 reports or persisted artifacts.

**Why keep Random Forest when LSTM has better RMSE?**  
LSTM won validation accuracy, but Random Forest had a better late-warning-sensitive NASA score on the final benchmark and direct TreeSHAP explanations. Promotion is held for risk review.

**Why show model disagreement?**  
It is an uncertainty signal. A large difference shows that two model families interpret the history differently and the case needs review.

**What is server-side pagination?**  
The API filters and sorts the full set, then returns one requested page instead of sending the entire fleet to the browser.

**How do you stop the LLM inventing RUL?**  
RUL comes from the persisted model, is placed in verified evidence, and is restored and validated after generation. The generator is never its source.

**What if an artifact is missing?**  
Health becomes degraded and dependent endpoints return 503 instead of fabricated data.

**How would you scale it?**  
Separate training and serving, publish approved models to a registry, stream sensors, run stateless API replicas, cache retrieval, queue generation, and monitor drift and outcomes.

**Biggest limitation?**  
FD001 is simulated with anonymized sensors, one operating condition and one degradation mode. This is engineering decision support, not an airworthiness system.

## 13. One-minute project explanation

> AeroGuard AI is an end-to-end predictive-maintenance application built on NASA C-MAPSS turbofan data. I created reproducible ingestion and validation, constructed train and test RUL correctly, and prevented leakage by splitting complete engines. I compared classical models with MLflow, then built a 30-cycle LSTM and TreeSHAP explanations. LSTM won validation RMSE, while Random Forest stayed the deployment champion because of its better late-warning risk score and direct explainability. I built a grounded RAG layer that combines verified model evidence with cited guidance without allowing the generator to change RUL. Finally, I exposed the artifacts through FastAPI and a fleet dashboard with validation, health checks, pagination, tests, CI and a cloud blueprint. It is a decision-support demonstration, not an airworthiness system.

## 14. Run Day 5

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_day5.ps1
```

macOS/Linux:

```bash
make app
```

Open `http://127.0.0.1:8000` for the dashboard and `http://127.0.0.1:8000/docs` for API documentation. Stop with `Ctrl+C`.
