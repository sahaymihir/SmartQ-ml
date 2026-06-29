# SmartQ ML Models

## Live Runtime Pieces

### Triage

- Runtime endpoint: `POST /predict`
- Live artifact directory: `triage_v3/model/`
- Status: production
- Type: trained XGBoost bundle

Tracked files:

- `triage_v3/model/triage_model_v3.pkl`
- `triage_v3/model/scaler_v3.pkl`
- `triage_v3/model/feature_cols_v3.pkl`

Retraining currently happens through `ml_service/auto_ml_pipeline_v3.py`.

### Specialty Routing

- Runtime endpoint: `POST /specialty`
- Live implementation: `ml_service/specialty_hybrid.py`
- Status: production
- Type: rule-based hybrid

`specialty_v2/training/train_specialty_v2.py` is still only a placeholder and does not back the live API.

### Test Recommendations

- Runtime endpoint: `POST /test-recommendations`
- Live implementation: rule engine in `ml_service/main.py`
- Status: production baseline
- Type: rule-based

`tests_v1/training/train_tests_v1.py` is still only a placeholder and does not back the live API.

## Local-Only Data

Keep datasets and generated outputs local:

- `ml_service/data/`
- `ml_service/src/`
- `ml_service/reports/`
- `ml_service/results/`
- `*/training/datasets/`
