# SmartQ ML Service

FastAPI service used by SmartQ for triage scoring, length-of-stay prediction, specialty
routing, and test recommendations — with self-monitoring of its own predictions.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design rationale.

## Endpoints

- `POST /predict` — KTAS priority 1–5 from intake fields (XGBoost triage bundle + clinical guardrails).
- `POST /predict-los` — expected ED length of stay in hours (XGBoost regressor).
- `POST /specialty` — rule-based specialty routing for doctor matching.
- `POST /test-recommendations` — rule-based diagnostic test suggestions.
- `POST /outcomes` — attach ground-truth labels to logged predictions.
- `GET /monitoring` — live accuracy, error, latency, and PSI drift dashboard.
- `GET /playground` — clinical testing UI.
- `GET /health` — health check.
- `GET /docs` — OpenAPI.

## Layout

```text
app/
├── main.py          # FastAPI app + triage/LOS inference + guardrails + recommendations
├── monitoring.py    # SQLite prediction logging + PSI drift
└── specialty.py     # rule-based specialty router
training/
├── train_triage.py  # triage classifier training pipeline
├── train_los.py     # LOS regressor training
└── evaluate.py      # held-out evaluation
scripts/
└── seed_monitoring.py   # shadow backtest to populate the monitoring store
models/               # tracked artifacts (triage_v3, los_v1, reference_stats.json)
static/               # dashboard + playground UI
data/                 # local-only datasets (gitignored)
```

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Retraining and evaluation

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python training/train_triage.py
python training/train_los.py
python training/evaluate.py
python scripts/seed_monitoring.py   # populate the monitoring dashboard
```

## Docker

```bash
docker build -t smartq-ml-service .
docker run --rm -p 8000:8000 smartq-ml-service
```

## Data policy

Datasets under `data/*.csv` stay local and are gitignored (Kaggle/hackathon data). Trained
model artifacts under `models/` **are** tracked — this is an ML showcase repo.

## Backend notes

- The Node backend actively calls `/predict` and `/specialty`, and degrades to rules if the
  ML service is down.
- Engineered triage features are recomputed inside the service so runtime inference stays
  aligned with the saved v3 bundle (no train/serve skew).
</content>
</invoke>
