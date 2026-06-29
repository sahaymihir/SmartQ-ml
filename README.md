# SmartQ ML Service

FastAPI service used by SmartQ for triage scoring, specialty routing, and test recommendations.

## What Is Actually Live

1. `POST /predict`
Uses the tracked XGBoost triage bundle in `models/triage_v3/model/`.

2. `POST /specialty`
Uses `specialty_hybrid.py`, a rule-based specialty router with symptom normalization and routing logic.

3. `POST /test-recommendations`
Uses the rule-based recommendation engine inside `main.py`.

Only the triage bundle is a tracked trained model right now.

## Folder Guide

```text
ml_service/
├── main.py
├── specialty_hybrid.py
├── auto_ml_pipeline_v3.py
├── evaluate_saved_model.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── models/
│   ├── README.md
│   ├── triage_v3/
│   │   ├── model/                 # tracked production artifacts
│   │   └── training/
│   ├── specialty_v2/
│   │   └── training/              # placeholder only
│   └── tests_v1/
│       └── training/              # placeholder only
├── data/                          # local-only datasets
├── src/                           # local-only specialty experiment datasets
├── reports/                       # local-only evaluation output
└── results/                       # local-only checkpoints
```

## Endpoints

- `POST /predict`: KTAS priority prediction from structured intake fields.
- `POST /specialty`: specialty routing for doctor matching.
- `POST /test-recommendations`: rule-based diagnostic test suggestions.
- `GET /health`: health check.

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Retraining And Evaluation

The real triage retraining pipeline is `auto_ml_pipeline_v3.py`.

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python auto_ml_pipeline_v3.py
python evaluate_saved_model.py
```

The scripts under `models/specialty_v2/training/` and `models/tests_v1/training/` are placeholders and do not back the live API yet.

## Data Policy

These stay local and should not be committed:

- `ml_service/data/`
- `ml_service/src/`
- `ml_service/results/`
- `ml_service/reports/`
- `ml_service/models/**/training/datasets/`
- local logs, PID files, virtual environments, and caches

## Docker

```bash
docker build -t smartq-ml-service .
docker run --rm -p 8000:8000 smartq-ml-service
```

## Backend Notes

- The backend actively calls `/predict` and `/specialty`.
- The frontend also has a response model for `/test-recommendations`.
- Engineered triage features are recomputed inside the service so runtime inference stays aligned with the saved v3 bundle.
