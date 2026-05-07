# Using the system outside the US (Cairo, etc.)

## What changed

Two new artifact pairs trained on the **same US data, same labels**, but with all geographic features removed:

| Original (geo-aware)        | Geo-agnostic                              | F1 / accuracy             |
|-----------------------------|-------------------------------------------|---------------------------|
| `severity_model.pkl`        | `severity_model_geo_agnostic.pkl`         | 73 % → 68 % accuracy       |
| `traffic_st_model.pkl`      | `traffic_st_model_geo_agnostic.pkl`       | F1 0.811 → **0.813** (slightly better) |

The spatiotemporal model improved when the geographic crutch was removed. It can no longer memorize per-cell baselines, so it learned to lean on weather and infrastructure — features that exist identically in Cairo.

## How to enable it

```bash
# 1) Re-run the standard pipeline first if you haven't already:
python spark/data_cleaning.py
python spark/feature_engineering.py
python spark/traffic_labeling.py
python spark/spatiotemporal_labeling.py
python models/train_model.py
python models/train_traffic_spatiotemporal.py

# 2) Train the geo-agnostic versions (this is the new step):
python models/train_geo_agnostic.py
# writes:
#   models/severity_model_geo_agnostic.pkl
#   models/scaler_severity_geo_agnostic.pkl
#   models/severity_features_geo_agnostic.pkl
#   models/traffic_st_model_geo_agnostic.pkl
#   models/traffic_st_features_geo_agnostic.pkl

# 3) Restart the API — it auto-detects the geo-agnostic artifacts:
python apis/main.py
```

The API loads both pairs at startup. Every `/predict` request is routed automatically based on whether `(latitude, longitude)` falls inside the continental-US bounding box (lat 24-49.5, lng -125 to -66). Response includes `region: "us" | "outside_us"` and `models_used` so the dashboard can label which models produced the answer.

## Unit conversion

Send `units: "metric"` in the request body to use °C, km/h, mm, km, hPa. The API converts to the imperial units the models were trained on before predicting. The default is `imperial` so the existing US flow is unchanged.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "latitude":  30.0444,
  "longitude": 31.2357,
  "hour": 17,
  "weather_condition": "Clear",
  "road_type": "Local",
  "units": "metric",
  "temperature": 32,
  "wind_speed": 15,
  "visibility": 16,
  "precipitation": 0,
  "pressure": 1013
}'
```

## Dashboard

A new **Units** dropdown sits below "Road Type". Imperial keeps the existing US-style labels; Metric flips them to (km, °C, mm, km/h) and the request body sends `units: "metric"` automatically. The numeric values stay where the user typed them — flip the toggle, then re-enter in the new unit.

## Sanity checks tested on your data

```
Cairo (30.0444, 31.2357), Clear, Local Road
   12:00 noon              →  Traffic LOW    (P[L,M,H] = 0.64, 0.32, 0.03)
   17:00 rush hour         →  Traffic MEDIUM (P[L,M,H] = 0.29, 0.49, 0.22)
   17:00 + 32°C metric     →  Traffic HIGH   (P[L,M,H] = 0.12, 0.36, 0.52)
```

Same coordinates, different times and conditions → different predictions. The temporal and weather signals transfer even though the spatial signal doesn't.

## Honest limitations

* The geo-agnostic model was trained on **US accident records**. It captures *generic* patterns of how time-of-day, weather, and infrastructure affect accident likelihood — not Cairo-specific traffic dynamics. Cairo's microbus ecosystem, ring-road behavior, informal merging, and local rush-hour patterns aren't in this model.
* For genuine Cairo predictions, you'd need Cairo data — either CAPMAS road-accident statistics, the WAZE Traffic Data Sharing Program, or live-traffic logging from Google/HERE APIs over time. See `docs/SPATIOTEMPORAL_DESIGN.md` § 8 for the upgrade path.
* The model's `outside_us` prediction is therefore "best-effort transferable estimate, not validated on Egyptian data." Communicate this clearly to anyone reviewing the dashboard. The API response includes a `note` field with this caveat for non-US requests so the frontend can surface it.
