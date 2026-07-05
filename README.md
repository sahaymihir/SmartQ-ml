# SmartQ ML Service

FastAPI service behind **SmartQ**, a hospital emergency-department queue app (Android +
Node/Express + MongoDB). It turns a patient's intake — free-text symptoms and vitals — into:

- a **triage priority** (KTAS 1–5) with clinical safety guardrails,
- an **expected length of stay** in hours,
- a **routed specialty** for doctor matching,
- a **diagnostic test panel**, and
- a **queue/route assignment** given live queue state,

and it **monitors its own predictions** in production (live accuracy, error, latency, drift).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design rationale and trade-offs.

---

## Table of contents

- [Endpoints](#endpoints)
- [API reference](#api-reference)
  - [`POST /predict`](#post-predict--triage-priority)
  - [`POST /predict-los`](#post-predict-los--length-of-stay)
  - [`POST /specialty`](#post-specialty--specialty-routing)
  - [`POST /test-recommendations`](#post-test-recommendations--diagnostic-tests)
  - [`POST /patient-flow`](#post-patient-flow--full-orchestration)
  - [`POST /outcomes`](#post-outcomes--ground-truth-feedback)
  - [`GET /monitoring/summary`](#get-monitoringsummary--dashboard-json)
- [Input handling & defaults](#input-handling--defaults)
- [Layout](#layout)
- [Local run](#local-run)
- [Retraining and evaluation](#retraining-and-evaluation)
- [Docker](#docker)
- [Data policy](#data-policy)
- [Backend notes](#backend-notes)

---

## Endpoints

| Method | Path                     | Purpose |
|--------|--------------------------|---------|
| `POST` | `/predict`               | KTAS priority 1–5 (XGBoost triage bundle + clinical guardrails). |
| `POST` | `/predict-los`           | Expected ED length of stay in hours (XGBoost regressor). |
| `POST` | `/specialty`             | Hybrid symptom→specialty routing for doctor matching. |
| `POST` | `/test-recommendations`  | Rule-based diagnostic test suggestions. |
| `POST` | `/patient-flow`          | One call that runs safety → triage → specialty → queue routing → tests. |
| `POST` | `/outcomes`              | Attach ground-truth labels to logged predictions. |
| `GET`  | `/monitoring`            | Live accuracy / error / latency / PSI-drift dashboard (HTML). |
| `GET`  | `/monitoring/summary`    | The same metrics as JSON. |
| `GET`  | `/playground`            | Clinical testing UI (HTML). |
| `GET`  | `/health`                | Health check. |
| `GET`  | `/`                      | Service metadata. |
| `GET`  | `/docs`                  | OpenAPI / Swagger UI. |

All request bodies are JSON. All clinical fields are **optional** — missing values are filled
from training-set medians (numeric) and modes (categorical); see
[Input handling & defaults](#input-handling--defaults).

---

## API reference

### `POST /predict` — triage priority

Predicts the KTAS acuity class from vitals, complaint, and demographics.

**Request** (all fields optional; ~40 accepted — the clinically useful ones shown):

```json
{
  "age": 67,
  "sex": "M",
  "spo2": 89,
  "respiratory_rate": 26,
  "heart_rate": 118,
  "systolic_bp": 92,
  "diastolic_bp": 60,
  "temperature_c": 38.9,
  "gcs_total": 14,
  "pain_score": 7,
  "news2_score": 8,
  "chief_complaint_system": "respiratory",
  "mental_status_triage": "alert"
}
```

**Response:**

```json
{
  "priority_class": 2,
  "confidence": 0.7421,
  "low_confidence": false,
  "recommendation": "Emergency — seen within 15 minutes",
  "all_class_probs": {"1": 0.12, "2": 0.74, "3": 0.10, "4": 0.03, "5": 0.01}
}
```

- `low_confidence` is `true` when the top-class probability is below **0.60** — a
  human-in-the-loop flag, not an error.
- `/predict` itself does **not** apply the safety guardrails — those live in `/patient-flow`.
  Call `/patient-flow` if you want the guardrailed class.

### `POST /predict-los` — length of stay

Predicts `ed_los_hours` (how long the patient will occupy the ED) at intake.

**Request** (sparse clinical payload, all optional):

```json
{"age": 67, "triage_acuity": 2, "chief_complaint_system": "respiratory",
 "spo2": 89, "heart_rate": 118, "news2_score": 8, "num_comorbidities": 3}
```

**Response:**

```json
{
  "predicted_hours": 4.87,
  "band_low": 3.72,
  "band_high": 6.02,
  "stay_category": "medium",
  "model_mae_hours": 1.146
}
```

- `band_low`/`band_high` are a ±MAE confidence band around the point estimate.
- `stay_category`: `short` (<2h), `medium` (2–5h), `long` (>5h).
- If the LOS bundle fails to load at startup, this endpoint returns **503**; the rest of the
  service keeps working.

### `POST /specialty` — specialty routing

Hybrid symptom→specialty engine. Normalises free-text symptoms (lower-casing, spell-correction,
repeated-letter collapse), scores specialty signals, and maps the winner to a **staffed route**.

**Request:**

```json
{"symptoms": "crushing chest pian and left arm numbness", "age": 58, "sex": "M",
 "pain_score": 8}
```

**Response:**

```json
{
  "primarySpecialist": "Cardiology",
  "routedSpecialty": "Cardiology",
  "confidence": 0.81,
  "lowConfidence": false,
  "normalizedSymptoms": "crushing chest pain and left arm numbness",
  "extractedSignals": ["chest pain", "arm numbness"],
  "alternatives": [
    {"specialist": "Neurology", "routedSpecialty": "Neurology", "confidence": 0.24,
     "matchedSignals": ["numbness"]}
  ],
  "reasoning": "Primary clinical fit is Cardiology based on signals like chest pain...",
  "modelSource": "specialty_hybrid_v1"
}
```

- `primarySpecialist` is the clinical best-fit; `routedSpecialty` collapses specialties with no
  dedicated staffed queue into **General OPD** (see `SPECIALTY_ROUTE_MAP`).
- `lowConfidence` is `true` below **0.58**, signalling manual review / doctor override.

### `POST /test-recommendations` — diagnostic tests

Rule-based test panel. Baseline engine (`rule_based_v1`) that a supervised model can hot-swap
later by replacing `generate_test_recommendations()`.

**Request** (same clinical fields as `/predict`, all optional):

```json
{"priority_class": 2, "chief_complaint_system": "cardiac", "age": 58,
 "temperature_c": 37.2, "pain_score": 8}
```

**Response:**

```json
{
  "recommendations": [
    {"test": "ABG (arterial blood gas)", "rationale": "Critical patient - assess oxygenation and acid-base", "urgency": "immediate"},
    {"test": "12-lead ECG", "rationale": "Rule out ST-elevation MI and arrhythmia", "urgency": "immediate"},
    {"test": "Troponin I/T", "rationale": "Biomarker for myocardial injury", "urgency": "immediate"}
  ],
  "source": "rule_based_v1",
  "low_confidence": false
}
```

Rules stack by complaint system, then layer on fever (≥38.5 °C), severe pain (≥7), paediatric
(≤15), elderly (>65), and critical-triage (class ≤2) panels; duplicates are removed. An empty
match falls back to a standard CBC + metabolic panel.

### `POST /patient-flow` — full orchestration

The one endpoint the app can call to get everything at once. Runs, in order:

1. **Symptom normalisation** + chief-complaint inference from free text.
2. **Safety rules** (`evaluate_safety_rules`) — clinical red flags that can force the priority up.
3. **Triage model** (`/predict` internally), then **guardrail resolution**: the strongest forced
   class wins only if it's *more* urgent than the model.
4. **Specialty routing** (`/specialty` internally).
5. **Queue assignment** — picks the best available route from live queue state, weighting wait
   time, queue length, staffing, and route preference (high-acuity patients weighted harder).
6. **Test recommendations**.

**Request:**

```json
{
  "symptoms": "can't breathe, chest tightness",
  "age": 71, "sex": "F", "spo2": 88, "respiratory_rate": 28, "heart_rate": 120,
  "gcs_total": 15,
  "availableRoutes": [
    {"route": "Pulmonology", "currentQueueLength": 3, "availableDoctors": 1, "avgWaitMinutes": 25},
    {"route": "General OPD", "currentQueueLength": 8, "availableDoctors": 2, "acceptsFallback": true}
  ]
}
```

**Response** (abridged) bundles `normalizedSymptoms`, `derivedChiefComplaintSystem`, a `safety`
list of matched rules, a `priority` block (model class **and** guardrailed class + `source`),
`specialty`, `queueAssignment`, and `tests`. `availableRoutes` is optional — omit it and the
clinically preferred route is returned directly.

**Safety rules** (each can force a priority class): critical/moderate hypoxia (SpO₂ ≤90 / ≤93),
severely/altered consciousness (GCS ≤8 / ≤12), possible stroke, loss of consciousness or seizure,
respiratory distress, cardiac red flag, and possible sepsis. Critical rules also override the
route to General OPD for immediate handling.

### `POST /outcomes` — ground-truth feedback

Attaches an observed label to a previously logged prediction, so the dashboard can compute live
accuracy / MAE.

```json
{"prediction_id": 142, "actual": 2}
```

`actual` is the observed triage class (1–5) or actual LOS in hours, depending on which endpoint
produced `prediction_id`. Returns `404` if the id is unknown.

### `GET /monitoring/summary` — dashboard JSON

Returns totals, per-endpoint counts, average latency, live triage accuracy, live LOS MAE, a
triage confidence histogram, the predicted-class distribution, and per-feature PSI drift. This is
the data the `/monitoring` HTML page renders.

---

## Input handling & defaults

Every clinical field is optional. Sparse payloads are made model-ready by
`frame_from_provided()` (shared by triage and LOS):

- **Numeric gaps** → training-set medians (`NUMERIC_DEFAULTS`).
- **Categorical gaps** → training-set modes (`CATEGORICAL_DEFAULTS`), kept semantically neutral.
- **Category aliases** → normalised (`"female"`→`"F"`, `"walkin"`→`"walk-in"`, …) and
  token-canonicalised, so slightly-off inputs still hit a valid encoder class instead of erroring.
- **Engineered vitals** (`shock_index`, `mean_arterial_pressure`, `pulse_pressure`,
  `spo2_resp_interaction`, `bmi`, `multi_risk_flag`, `age_group`, `shift`, `arrival_season`, …)
  are **recomputed at inference** from the same logic used in training — no train/serve skew.
- `/predict` rejects unknown fields (`extra="forbid"`); `/predict-los`, `/specialty`, and
  `/test-recommendations` ignore them (`extra="ignore"`).

---

## Layout

```text
app/
├── main.py          # FastAPI app: triage/LOS inference, guardrails, tests, patient-flow orchestration
├── monitoring.py    # SQLite prediction logging + PSI drift + summary
└── specialty.py     # hybrid symptom→specialty router (normalisation + signal scoring)
training/
├── train_triage.py  # triage classifier training pipeline
├── train_los.py     # LOS regressor training (+ reference_stats.json snapshot)
└── evaluate.py      # held-out evaluation
scripts/
└── seed_monitoring.py   # shadow backtest to populate the monitoring store
models/               # tracked artifacts (triage_v3/, los_v1/, reference_stats.json)
static/               # dashboard + playground UI (index.html, monitoring.html, app.js, styles.css)
data/                 # local-only datasets (gitignored)
```

---

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open <http://localhost:8000/playground> (clinical UI),
<http://localhost:8000/monitoring> (dashboard), or <http://localhost:8000/docs> (OpenAPI).

Quick smoke test:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"age":67,"spo2":89,"respiratory_rate":26,"heart_rate":118,"chief_complaint_system":"respiratory"}'
```

---

## Retraining and evaluation

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python training/train_triage.py     # triage classifier
python training/train_los.py        # LOS regressor + reference_stats.json drift snapshot
python training/evaluate.py         # held-out metrics
python scripts/seed_monitoring.py   # shadow-backtest the test split into the monitoring store
```

`scripts/seed_monitoring.py` replays the held-out test split through the **live** models and
records the true labels (tagged `source='backtest'`, never disguised as production traffic), so
the dashboard shows real numbers immediately and confirms serving matches training.

`app/monitoring.py` has a stdlib self-check: `python -m app.monitoring` inserts, labels, and
asserts the summary numbers.

---

## Docker

```bash
docker build -t smartq-ml-service .
docker run --rm -p 8000:8000 smartq-ml-service
```

---

## Data policy

Datasets under `data/*.csv` stay local and are gitignored (Kaggle/hackathon data). Trained model
artifacts under `models/` **are** tracked — this is an ML showcase repo.

---

## Backend notes

- The Node backend actively calls `/predict` and `/specialty`, and degrades to rules if the ML
  service is down.
- Engineered triage features are recomputed inside the service so runtime inference stays aligned
  with the saved v3 bundle (no train/serve skew).
- Prediction logging is wrapped so it can never break inference — a monitoring-store failure is
  swallowed and logged, and the prediction still returns.
