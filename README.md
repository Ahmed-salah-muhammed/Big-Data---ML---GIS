<div align="center">

# TrafficIQ

**Real-Time Spatiotemporal Prediction System for Roads — with Cross-Region Transfer to Cairo**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-EB6E2F)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9-199900?logo=leaflet&logoColor=white)](https://leafletjs.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Status](https://img.shields.io/badge/status-final--project-success)]()

End-to-end ML pipeline that predicts accident severity and traffic congestion in real time from a single map click — trained on 451k US accident records, deployable to any city worldwide via a geo-agnostic model variant.

[Features](#features) · [Quick Start](#quick-start) · [API](#api-reference) · [Results](#results) · [Documentation](#documentation) · [Team](#team)

</div>

---

## Why this project exists

Real-time traffic and accident prediction is a foundational service for any modern Intelligent Transportation System. Most public datasets are North American; models trained on them silently extrapolate when pointed at Cairo or any non-US city. **TrafficIQ solves both problems**: an empirical spatiotemporal labeling scheme that defeats target leakage, plus a geo-agnostic model variant trained without any geographic features so it transfers cleanly to other regions.

The project also serves as a documented case study in target leakage. The first model achieved **99.98 % accuracy** — and that number was meaningless, because the label was a deterministic rule over the same features the model received. The reformulation, the honest **0.81 macro-F1** that resulted, and the 13 bugs found along the way are all written up in detail.

---

## Features

- 🚦 **Two prediction tasks** — binary accident severity (high/low risk) and 3-class traffic level (Low / Medium / High)
- 🌍 **Cross-region transfer** — geo-agnostic model variant works in Cairo, Riyadh, Mumbai, anywhere
- 🧠 **Honest, leakage-free labels** — empirical density per `(cell, hour)` bucket, not hand-coded rules
- ⚡ **Fast** — < 100 ms per prediction
- 🛡️ **Three-tier graceful degradation** — geo-aware → geo-agnostic → simulator; the API never returns a 500
- 🔁 **Unit-aware** — accepts metric or imperial inputs, converts server-side
- 🗺️ **Single-file Leaflet.js dashboard** — no build step, opens directly in any browser
- 📚 **Reproducible** — three sequential pandas modules, one entry point each

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER (Browser, click on map)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Leaflet.js  Dashboard     │   ← dashboard/index.html
              │   (single-file, vanilla JS) │
              └──────────────┬──────────────┘
                             │  POST /predict
              ┌──────────────▼──────────────┐
              │     FastAPI  Service        │   ← apis/main.py
              │  /predict · /weather · /route
              └──┬─────────┬────────┬───────┘
                 │         │        │
        ┌────────▼─┐  ┌────▼────┐ ┌─▼─────────────┐
        │ Severity │  │ Traffic │ │ Geo-agnostic  │
        │  XGBoost │  │ XGBoost │ │   variants    │
        │ 28 feat. │  │ 22 feat.│ │ for non-US    │
        └──────────┘  └─────────┘ └───────────────┘
                  trained offline · models/*.pkl
```

For the visual workflow infographic see `docs/figures/01_workflow.png` or page 10 of the final report.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-org>/TrafficIQ.git
cd TrafficIQ
pip install -r requirements.txt
```

> **Note on `requirements.txt`** — the file is plain UTF-8. Earlier versions were saved as UTF-16 BOM which `pip` couldn't parse; that's bug #1 in `BUGS_FIXED.md`.

### 2. Get the dataset

Download the [US Accidents 2016–2023 dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle and place it at:

```
data/raw/dirty_dataset.csv
```

### 3. Run the pipeline (in order)

```bash
# Stage 1 — Clean raw data
python spark/data_cleaning.py
#   → data/processed/cleaned_data.csv

# Stage 2 — Feature engineering
python spark/feature_engineering.py
#   → data/processed/featured_data.csv

# Stage 3a — Rule-based labeling (legacy, kept for reference)
python spark/traffic_labeling.py
#   → data/processed/final_dataset.csv

# Stage 3b — Empirical spatiotemporal labeling
python spark/spatiotemporal_labeling.py
#   → data/processed/spatiotemporal_dataset.csv
#   → data/processed/cell_baseline_lookup.csv
#   → data/processed/spatiotemporal_config.json

# Train all models
python models/train_model.py                       # severity + legacy traffic
python models/train_traffic_spatiotemporal.py      # the honest 0.811 F1 model
python models/train_geo_agnostic.py                # Cairo-ready variants
```

Total runtime on a modern laptop: **about 5 minutes** end-to-end.

### 4. Serve

```bash
python apis/main.py
# Listening on http://localhost:8000
```

Then open `dashboard/index.html` directly in any browser.

---

## API Reference

The FastAPI service exposes five JSON endpoints. All are stateless and CORS-permissive.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`                      | Health check; reports which models loaded |
| `GET`  | `/weather?lat=&lng=`     | Server-side proxy to Open-Meteo for live conditions |
| `GET`  | `/route?start_lat=&start_lng=&end_lat=&end_lng=` | OSRM driving route with fallback mirror |
| `POST` | `/predict`               | Single-point prediction; auto-routes by US-bbox region |
| `POST` | `/predict-spatiotemporal`| Direct access to the 3-class traffic model |

### Example: predict at El Tahrir Square, Cairo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 30.0444,
    "longitude": 31.2357,
    "hour": 17,
    "weather_condition": "Clear",
    "road_type": "Local",
    "units": "metric",
    "temperature": 32,
    "wind_speed": 15,
    "visibility": 16,
    "precipitation": 0
  }'
```

```json
{
  "region": "outside_us",
  "models_used": {
    "severity": "geo_agnostic",
    "traffic":  "geo_agnostic_spatiotemporal"
  },
  "traffic_level": "Medium",
  "spatiotemporal": {
    "traffic_level": "Medium",
    "probabilities": { "Low": 0.29, "Medium": 0.49, "High": 0.22 }
  },
  "risk_level": "High",
  "accident_probability": 0.677,
  "details": {
    "road_type":   "Local",
    "is_rush_hour": true,
    "units_received": "metric"
  },
  "note": "Models trained on US Accidents 2016-2023..."
}
```

### Region routing

The `/predict` endpoint automatically routes requests by location:

- Inside the **continental-US bounding box** (lat 24-49.5, lng -125 to -66) → **geo-aware** models (use cell baselines, sharper accuracy)
- Anywhere else → **geo-agnostic** models (transferable; weather + infrastructure only)

### Unit conversion

Send `"units": "metric"` and the API converts °C → °F, km/h → mph, mm → in, km → mi, hPa → inHg before predicting.

---

## Results

### Headline metrics (5-fold CV on 451,837 records)

| Model | Variant | Metric | Score |
|-------|---------|--------|-------|
| **Severity** | Geo-aware | Accuracy | **0.730** |
| | | Recall (high-risk class) | **0.776** |
| Severity | Geo-agnostic | Accuracy | 0.680 |
| **Traffic** ⚠ | Original (rule-based) | Accuracy | 0.9998 ← misleading, target leakage |
| **Traffic** | Spatiotemporal (geo-aware) | F1-macro | **0.811 ± 0.003** |
| **Traffic** | Spatiotemporal (geo-agnostic) | F1-macro | **0.813 ± 0.002** |

### What the spatiotemporal model gets right

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Low    | 0.71 | 0.98 | 0.83 |
| Medium | 0.92 | 0.59 | 0.72 |
| **High** | **0.98** | **0.97** | **0.97** |

The High class — the *actionable* one for traffic warnings — is essentially solved. The Low/Medium boundary is structurally fuzzy (both classes contain `count = 1` buckets) and would tighten under a binary `count = 1 vs count ≥ 2` reformulation.

### Cairo cross-region sanity check (geo-agnostic model, El Tahrir Sq)

```
   12:00 noon, Clear              →  Low traffic      (P[L,M,H] = 0.64, 0.32, 0.03)
   17:00 rush hour                →  Medium traffic   (P[L,M,H] = 0.29, 0.49, 0.22)
   17:00 + 32°C metric inputs     →  High traffic     (P[L,M,H] = 0.12, 0.36, 0.52)
```

Same coordinates, three different time/condition combos → three different predictions. The model is genuinely responsive in Cairo, despite never seeing Egyptian data.

---

## Project Structure

```
TrafficIQ/
├── spark/                              # ETL pipeline (pandas)
│   ├── data_cleaning.py                  Stage 1
│   ├── feature_engineering.py            Stage 2
│   ├── traffic_labeling.py               Stage 3a — legacy rule-based labels
│   └── spatiotemporal_labeling.py        Stage 3b — empirical density labels
│
├── models/                             # Training scripts + saved artifacts
│   ├── train_model.py                    severity + legacy traffic
│   ├── train_traffic_spatiotemporal.py   the honest 0.811 F1 model
│   ├── train_geo_agnostic.py             Cairo-ready variants
│   └── XGBoost.py                        deprecated stub (was source of leakage)
│
├── apis/
│   └── main.py                         # FastAPI service (5 endpoints)
│
├── dashboard/
│   └── index.html                      # Leaflet + vanilla JS, opens directly
│
├── docs/
│   ├── SPATIOTEMPORAL_DESIGN.md          full design rationale
│   ├── CROSS_REGION_USAGE.md             how to deploy outside the US
│   └── figures/                          11 PNG figures used in the report
│
├── data/                               # git-ignored
│   ├── raw/dirty_dataset.csv             ← put Kaggle dataset here
│   └── processed/                        outputs of the ETL pipeline
│
├── BUGS_FIXED.md                       # 13 issues found and resolved
├── CLAUDE.md                           # project map for future devs
├── requirements.txt
├── README.md                           # ← you are here
└── LICENSE
```

---

## Documentation

The project ships with four detailed companion documents:

| Document | What's in it |
|----------|--------------|
| [`docs/SPATIOTEMPORAL_DESIGN.md`](docs/SPATIOTEMPORAL_DESIGN.md) | Full design rationale for the empirical-density labeling scheme, including the cell-size sensitivity analysis and per-class structure |
| [`docs/CROSS_REGION_USAGE.md`](docs/CROSS_REGION_USAGE.md) | How to deploy outside the US; what the geo-agnostic model captures and what it doesn't |
| [`BUGS_FIXED.md`](BUGS_FIXED.md) | Trace of all 13 issues found in the code review, with severity tier, reproduction steps, and the patch that fixed each |
| [`CLAUDE.md`](CLAUDE.md) | High-level project map, written for someone touching the codebase for the first time |
| `TrafficIQ_Final_Report.pdf` | 27-page formal research-article report with all figures and references |

---

## Tech Stack

- **Python 3.12** · pandas 3.0 · numpy 2.4 · scikit-learn 1.8 · XGBoost 3.x · joblib
- **FastAPI 0.136** · uvicorn · pydantic · httpx
- **Leaflet.js 1.9** · vanilla JavaScript (no build step) · Tailwind CSS via CDN
- **Open-Meteo** (live weather) · **OSRM** (routing) — both server-side proxied
- **Matplotlib** for chart generation in the report

---

## Performance Notes

- Pipeline end-to-end: ≈ **5 minutes** on a modern laptop (Apple M2 / Intel i7)
- Single `/predict` request: **< 100 ms** including all three model invocations
- Spatiotemporal labeling step is vectorized — about **50× faster** than the original `df.apply(axis=1)` implementation (1.27 s for 100k rows)
- Geo-agnostic model uses **17 features** vs 22 in the geo-aware variant; smaller and faster while scoring marginally better F1

---

## Reproducibility Checklist

Before delivering anything, verify:

- [ ] `file requirements.txt` reports `Unicode text, UTF-8 text` (not UTF-16)
- [ ] `pip install --dry-run -r requirements.txt` succeeds
- [ ] `python -m py_compile spark/*.py models/*.py apis/main.py` passes
- [ ] `python -W error::UserWarning -c "import apis.main"` loads with zero warnings
- [ ] `/predict` end-to-end works for both a US point and Cairo

All five checks have been validated as part of the project handoff.

---

## Known Limitations

- Models are **trained on US data**. The geo-agnostic variant captures generic time-and-condition patterns, not Cairo-specific traffic dynamics (microbus stops, ring-road merging, informal flow).
- The spatiotemporal label is **structurally ambiguous** between Low and Medium classes; a binary reformulation would lift F1 toward 0.90.
- No `Date` column is preserved through cleaning, so **lag features** (count yesterday at this hour) are unavailable. Adding date preservation is the highest-leverage improvement remaining.
- Cell baselines are computed from the full dataset before train/test split — a stricter cold-start spatial holdout would be a more rigorous test.

See `docs/SPATIOTEMPORAL_DESIGN.md` § 8 for the full upgrade path.

---

## Team

Final project for the **Geographic Information Systems (GIS) Track**, 9-Month Professional Programme at the **Information Technology Institute (ITI)**, Ministry of Communications and Information Technology, Egypt.

| Name | Role |
|------|------|
| **Abdulrahman Omar** | Data pipeline · Feature engineering |
| **Muhammed Ashraf**  | Modeling · Cross-region transfer |
| **Ahmed Salah**      | API service · Dashboard · Report |

---

## License

Released under the [MIT License](LICENSE). The US Accidents dataset is owned by its original authors; please cite [Moosavi et al., 2019](https://arxiv.org/abs/1906.05409) when using it.

---

## Citing this work

```bibtex
@misc{trafficiq2026,
  title  = {TrafficIQ: A Real-Time Spatiotemporal Prediction System for Roads
            with Cross-Region Transfer to Cairo},
  author = {Omar, Abdulrahman and Ashraf, Muhammed and Salah, Ahmed},
  year   = {2026},
  note   = {GIS Track Final Project, Information Technology Institute (ITI), Egypt}
}
```

---

<div align="center">

Built with care at **ITI · Cairo · 2026** 🇪🇬

</div>
