# SmartQ ML Service — Architecture & Design Notes

The ML brain behind **SmartQ**, a hospital emergency-department queue app
(Android + Node/Express + MongoDB). This service turns a patient's intake — symptoms
and vitals — into a triage priority, an expected length of stay, a routed specialty, and
a recommended test panel, and **monitors its own predictions** in production.

This document is the "why", not just the "what". It's meant to be read top-to-bottom.

---

## 1. System at a glance

```
 Patient intake (symptoms + vitals)
            │
            ▼
 ┌──────────────────────────────────────────────┐
 │  FastAPI service (main.py)                     │
 │                                                │
 │  1. Safety guardrails  ── clinical red-flag    │
 │     (rules)               rules can override   │
 │            │              the model downward    │
 │            ▼                                    │
 │  2. Triage model  ──►  KTAS priority 1–5        │  XGBoost classifier
 │     (/predict)         + confidence             │  85.2% acc, weighted ROC-AUC
 │            │                                    │
 │            ▼                                    │
 │  3. LOS model     ──►  expected hours in ED     │  XGBoost regressor
 │     (/predict-los)     + stay band              │  MAE 1.15h (−40% vs baseline)
 │            │                                    │
 │            ▼                                    │
 │  4. Specialty router + test recommender        │  rule/signal engines
 │            │                                    │
 │            ▼                                    │
 │  every prediction ──► monitoring store (SQLite) │
 └──────────────────────────────────────────────┘
            │
            ▼
 /monitoring dashboard  ── live accuracy, error, latency, PSI drift
```

The Node backend calls `/predict` (and can call `/predict-los`) over HTTP and degrades
gracefully if the ML service is down — triage falls back to rules.

---

## 2. The two models

### Triage classifier (`triage_v3`) — *pre-existing*
- **Task:** predict KTAS acuity class 1–5 from vitals + complaint + demographics.
- **Data:** 80,000-row ED dataset, 40 selected features.
- **Pipeline** (`auto_ml_pipeline_v3.py`): application-aware feature engineering →
  RandomForest feature selection → label-encode + scale → SMOTE (only if imbalance > 2×) →
  train LightGBM / XGBoost / soft-voting ensemble → `RandomizedSearchCV` tuning →
  confidence thresholding (0.60) → adjacency error analysis.
- **Served by** `/predict`, with a **clinical guardrail layer**: 9 hard rules (critical
  hypoxia, GCS ≤ 8, stroke keywords, sepsis physiology, …) that can force the priority
  *up* regardless of what the model says. The model never gets the last word on safety.

### Length-of-stay regressor (`los_v1`) — *added in this iteration*
- **Task:** predict `ed_los_hours` — how long a patient will occupy the ED — at intake
  time. Drives capacity planning and informs the patient's wait estimate.
- **Why it existed waiting to be built:** `ed_los_hours` was already in the dataset but
  *discarded* by the triage pipeline (it's not a triage feature). It's a ready-made
  regression target with 80k real labels.
- **Model:** `XGBRegressor`, reusing the triage pipeline's engineered-vitals features
  (`add_application_features`) so training and inference compute features identically.
- **The metric that matters — beat the naive baseline:**

  | Predictor            | MAE (h) | RMSE (h) | R²    |
  |----------------------|--------:|---------:|------:|
  | Baseline (mean LOS)  | 1.926   | 2.436    | 0.000 |
  | **XGBoost regressor**| **1.146** | **1.583** | **0.578** |

  **→ 40.5% reduction in mean absolute error** over predicting the average for everyone.
- **Served by** `/predict-los`, returning predicted hours, a ±MAE confidence band, and a
  stay category (`short <2h | medium 2–5h | long >5h`).

---

## 3. The monitoring loop (what makes it a *system*, not a notebook)

A trained model in a `.pkl` is a class project. Knowing whether it still works in
production is the engineering part. So every prediction is observable:

- **Persistence** (`monitoring.py`, stdlib SQLite): each `/predict` and `/predict-los`
  call logs its inputs, output, confidence, and latency. Logging is wrapped so it can
  never break inference.
- **Ground-truth feedback** (`POST /outcomes`): when the real triage class or actual LOS
  becomes known, it's attached to the logged prediction — enabling live accuracy / MAE.
- **Drift detection** (PSI — Population Stability Index): at train time we snapshot the
  decile distribution of key vitals (`models/reference_stats.json`); at runtime we compare
  recent inputs against it. PSI < 0.10 = stable, 0.10–0.25 = minor shift, > 0.25 = retrain
  signal. This catches the silent failure mode where inputs drift away from training data
  and accuracy quietly rots.
- **Dashboard** (`/monitoring`): KPI cards (volume, live accuracy, live MAE, latency),
  triage confidence histogram, predicted-class distribution, and per-feature PSI bars.
- **Seeding** (`seed_monitoring.py`): an honest **shadow backtest** replays the held-out
  test split through the live models and records true labels, so the dashboard shows real
  numbers immediately. These rows are tagged `source='backtest'` — never disguised as
  production traffic. Replayed numbers match the trainers' held-out metrics (acc 0.852,
  LOS MAE 1.136h), confirming the serving path matches the training path.

---

## 4. Design decisions (and the trade-offs)

- **Tabular gradient boosting, not an LLM.** Triage and LOS are structured-data problems
  with 80k labeled rows. XGBoost is more accurate here, ~30ms latency, deterministic, and
  auditable — all of which matter clinically. An LLM would be slower, costlier, and harder
  to defend to a regulator. LLM/NLP is used only where it fits: normalising free-text
  symptoms.
- **Rules can override the model, never the reverse, on safety.** A model that's 99%
  accurate still misses 1-in-100; for "SpO₂ 88" that's unacceptable. Hard guardrails bound
  the failure mode.
- **Confidence thresholding (0.60).** Low-confidence triage predictions are flagged rather
  than trusted silently — a human-in-the-loop hook.
- **PSI over a heavier drift framework.** PSI is ~30 lines of numpy, industry-standard, and
  explainable in one sentence. No Prometheus/Grafana/Evidently — SQLite + a static page is
  enough for the scale and keeps the whole thing readable.
- **One feature-engineering function, two consumers.** Both training and inference call the
  same `add_application_features` / `apply_engineered_features` logic, so there's no
  train/serve skew on the engineered vitals (`shock_index`, `multi_risk_flag`, …).

---

## 5. What I'd do next

- Wire `/predict-los` into the Node backend's queue wait estimate (the LOS prediction feeds
  the patient-facing "expected wait").
- Replace the rule-based specialty router and test recommender with trained models once the
  app collects enough labeled outcomes (the training scaffolds already exist).
- Persist `/outcomes` from the live app (token completion already records actual wait /
  consult time in MongoDB) to compute drift and accuracy on **real** traffic, not just the
  backtest.
- Scheduled retraining triggered by a PSI drift alert.

---

## 6. Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # training; requirements.txt for serving only

python train_los_v1.py                     # train LOS model, prints baseline-vs-model metrics
python seed_monitoring.py                  # populate the monitoring store (shadow backtest)
uvicorn main:app --reload                  # serve

# then open:
#   /playground   — clinical testing UI
#   /monitoring   — model monitoring dashboard
#   /docs         — OpenAPI
```
