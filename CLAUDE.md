# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time accident probability and traffic congestion prediction system for road networks. It ingests the US Accidents (2016–2023) dataset (500k+ records), runs a sequential ETL pipeline, trains two XGBoost binary classifiers, and serves predictions via a FastAPI REST API with a Leaflet.js GIS dashboard.

See **BUGS_FIXED.md** for a full trace of issues found and resolved during the most recent refactor.

## Setup

```bash
pip install -r requirements.txt
```

The previous `requirements.txt` was saved as UTF-16 BOM, which `pip` cannot parse. It is now plain UTF-8 and includes `xgboost`, `httpx`, and `python-multipart` (which the code already imported but the file had silently omitted).

## Running the Pipeline (in order)

```bash
# Step 1 – clean raw data
python spark/data_cleaning.py
# reads:  data/raw/dirty_dataset.csv
# writes: data/processed/cleaned_data.csv

# Step 2 – engineer features
python spark/feature_engineering.py
# reads:  data/processed/cleaned_data.csv
# writes: data/processed/featured_data.csv

# Step 3 – generate labels  (single source of truth for labels)
python spark/traffic_labeling.py
# reads:  data/processed/featured_data.csv
# writes: data/processed/final_dataset.csv

# Step 4 – train both models
python models/train_model.py
# reads:  data/processed/final_dataset.csv
# writes (all under models/):
#   traffic_model.pkl   + scaler_traffic.pkl   + traffic_features.pkl
#   severity_model.pkl  + scaler_severity.pkl  + severity_features.pkl
```

## Running the API

```bash
python apis/main.py
# Listens on http://localhost:8000
# GET  /                – health/status (which models loaded)
# GET  /weather?lat&lng – Open-Meteo proxy
# GET  /route?...       – OSRM proxy with fallback mirror
# POST /predict         – single-point traffic + severity prediction
# POST /predict-route   – segment-by-segment prediction along a polyline
```

Open `dashboard/index.html` directly in a browser to use the GIS frontend. The dashboard's `CFG.API` is hard-coded to `http://localhost:8000`.

## Architecture

```
spark/data_cleaning.py
    → spark/feature_engineering.py
        → spark/traffic_labeling.py     (single source of truth for labels)
            → models/train_model.py     (two independent XGBClassifiers)
                → apis/main.py  ←→  dashboard/index.html
```

**`spark/` (ETL pipeline — pandas-based, not Spark despite the directory name)**

- `data_cleaning.py` — drops 21 columns, repairs timestamps, coerces booleans, fills missing weather values with per-airport / per-month medians, extracts `Hour` and `Month`.
- `feature_engineering.py` — derives `Road_Type` deterministically via regex on street names, generates `Road_Width(m)` / `Speed_Limit(mph)` (deterministic defaults — the previous version used `np.random.uniform` for missing streets, which silently injected noise into training and inference), `Is_RushHour` (6–9 AM / 4–7 PM), `Is_Night` (9 PM–5 AM, with a `Sunrise_Sunset` override when present), and numeric-encodes weather and road type.
- `traffic_labeling.py` — vectorized rules-based congestion score (+2 narrow road, +2 rush hour, +1 low speed limit, +1 daytime, +2 bad weather; threshold ≥ 4); sets `High_Risk_Accident=1` when raw `Severity ≥ 3`.

**`models/train_model.py`**

- Trains two independent `XGBClassifier` models, each with its own `StandardScaler`:
  - **Traffic model** — 14 features → binary `Traffic_Congestion`
  - **Severity model** — 28 features → binary `High_Risk_Accident`
- `scale_pos_weight` is computed from the train split per target.
- `StandardScaler.set_output(transform="pandas")` keeps feature names through the whole pipeline so prediction-time DataFrames don't trigger `X does not have valid feature names` warnings.
- The previous design wrapped XGBoost in `sklearn.multioutput.MultiOutputClassifier` (`models/XGBoost.py`); that triggered `UserWarning: \`sklearn.utils.parallel.delayed\` should be used with \`sklearn.utils.parallel.Parallel\`...` on modern scikit-learn. Two separate classifiers eliminate the warning, give per-target threshold control, and are simpler to reason about. `models/XGBoost.py` is now a deprecation stub.

**`apis/main.py`**

- FastAPI app; loads saved `.pkl` artifacts at startup.
- Three-tier graceful degradation per target:
  1. Per-target model (`traffic_model.pkl`, `severity_model.pkl`) — preferred.
  2. Legacy multi-output `xgboosst.pkl` — fallback for whichever target is missing.
  3. Deterministic simulator — final fallback so the dashboard still works for demos.
- Predictions feed DataFrames (not numpy arrays) into the scaler / model to preserve feature names and silence sklearn warnings.
- `/weather` and `/route` are server-side proxies that bypass browser CORS / mixed-content blocks. They prefer `httpx` and fall back to stdlib `urllib`.

**`dashboard/index.html`**

- Single-file frontend using Leaflet.js + Tailwind CSS.
- Clickable map sets lat/lon; form controls set weather and time inputs.
- Calls `POST /predict` and `POST /predict-route`, renders results inline with an API status indicator.
- `dashboard/index3.html` is an older standalone version kept for reference; the active dashboard is `index.html`.

## Key Domain Rules (encoded in pipeline)

| Concept             | Rule                                                |
|---------------------|-----------------------------------------------------|
| Rush hour           | 6–9 AM or 4–7 PM                                    |
| Night               | 9 PM – 5 AM                                         |
| Congested           | Score ≥ 4 (see `traffic_labeling.py`)               |
| High accident risk  | Original severity ≥ 3                               |
| Weather encoding    | Clear=0, Cloudy=1, Rain=2, Snow=3, Fog=4            |
| Road type encoding  | Local=1, Internal=2, Highway=3                      |

## Data & Models (git-ignored)

`data/` and `models/*.pkl` are excluded from version control. The raw dataset is the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) dataset from Kaggle and must be placed at `data/raw/dirty_dataset.csv` before running the pipeline.

## Conventions for Future Edits

- Labels live in `traffic_labeling.py` only — do not recompute them in training.
- The feature-name lists `TRAFFIC_FEATURES` and `SEVERITY_FEATURES` in `models/train_model.py` are the contract between training and serving. If you add or remove a feature, update the list in one place; the training script saves it to `*_features.pkl` and the API loads it from there.
- Always pass `pandas.DataFrame` (not raw `numpy` arrays) into a fitted `StandardScaler` or model to keep feature names attached.
- Despite the directory name `spark/`, this project is pure pandas. If you migrate to PySpark, keep the import surface (the `run_*` functions and their I/O paths) identical so callers don't break.
