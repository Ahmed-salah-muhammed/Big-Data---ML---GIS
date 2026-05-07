"""
TrafficIQ FastAPI server.

Endpoints
---------
GET  /                      — health/status
GET  /weather?lat&lng       — server-side proxy to Open-Meteo
GET  /route?start_lat&...   — server-side proxy to OSRM (with fallback mirror)
POST /predict               — single-point traffic + severity prediction
POST /predict-route         — segment-by-segment prediction along a polyline

Run
---
    python apis/main.py        # listens on http://0.0.0.0:8000
    # or:
    uvicorn apis.main:app --host 0.0.0.0 --port 8000 --reload

Notes
-----
* Loads the artifacts produced by `models/train_model.py`:
    - traffic_model.pkl  + scaler_traffic.pkl  + traffic_features.pkl
    - severity_model.pkl + scaler_severity.pkl + severity_features.pkl
* Falls back to the legacy multi-output `xgboosst.pkl` ONLY if a per-target
  artifact is missing — that keeps old deployments alive while the user
  retrains.
* If both real and legacy artifacts fail to load, deterministic simulators
  drive the API so the dashboard stays usable for demos.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional: httpx is faster, but we degrade to stdlib urllib if it's missing.
try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except Exception:
    _HAS_HTTPX = False


# ===========================================================================
# Region detection + unit conversion
# ===========================================================================
# Continental US bounding box, slightly padded. Outside this box we route
# requests to the geo-agnostic models, which were trained without lat/lng.
US_BBOX = {"lat_min": 24.0, "lat_max": 49.5, "lng_min": -125.0, "lng_max": -66.0}


def is_in_continental_us(lat: float, lng: float) -> bool:
    return (
        US_BBOX["lat_min"] <= lat <= US_BBOX["lat_max"]
        and US_BBOX["lng_min"] <= lng <= US_BBOX["lng_max"]
    )


def to_imperial(value: float | None, kind: str) -> float | None:
    """Convert a metric value to the imperial unit the model was trained on.
    `kind` is one of: 'temperature', 'wind_speed', 'precipitation',
    'visibility', 'pressure'.
    """
    if value is None:
        return None
    if kind == "temperature":  # °C → °F
        return value * 9.0 / 5.0 + 32.0
    if kind == "wind_speed":  # km/h → mph
        return value * 0.621371
    if kind == "precipitation":  # mm → in
        return value / 25.4
    if kind == "visibility":  # km → mi
        return value * 0.621371
    if kind == "pressure":  # hPa (mbar) → inHg
        return value * 0.02953
    return value


def normalize_units(payload: dict[str, Any]) -> dict[str, Any]:
    """If the request says `units == "metric"`, convert weather fields in place."""
    if str(payload.get("units", "imperial")).lower() != "metric":
        return payload
    payload = dict(payload)  # copy
    payload["temperature"] = to_imperial(payload.get("temperature"), "temperature")
    payload["wind_speed"] = to_imperial(payload.get("wind_speed"), "wind_speed")
    payload["precipitation"] = to_imperial(
        payload.get("precipitation"), "precipitation"
    )
    payload["visibility"] = to_imperial(payload.get("visibility"), "visibility")
    payload["pressure"] = to_imperial(payload.get("pressure"), "pressure")
    return payload


# ===========================================================================
# HTTP helper
# ===========================================================================
def _http_get_json(url: str, timeout: float = 8.0) -> dict[str, Any]:
    """GET `url` and return parsed JSON, with httpx → urllib fallback."""
    if _HAS_HTTPX:
        try:
            with httpx.Client(timeout=timeout) as c:
                return c.get(url).json()
        except Exception:
            pass  # try urllib
    req = urllib.request.Request(url, headers={"User-Agent": "TrafficIQ/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_post_json(url: str, body: str, timeout: float = 12.0) -> dict[str, Any]:
    """POST raw body (Overpass uses form-style 'data=...')."""
    if _HAS_HTTPX:
        try:
            with httpx.Client(timeout=timeout) as c:
                return c.post(
                    url,
                    content=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                ).json()
        except Exception:
            pass
    req = urllib.request.Request(
        url,
        data=body.encode(),
        headers={
            "User-Agent": "TrafficIQ/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ===========================================================================
# Live data fetchers — OSM (road type) + Open-Meteo (weather)
# ===========================================================================
# OSM `highway=*` tag → our 3-class Road_Type
_OSM_HIGHWAY_TO_TYPE = {
    # Highway-class
    "motorway": "Highway", "trunk": "Highway", "primary": "Highway",
    "motorway_link": "Highway", "trunk_link": "Highway", "primary_link": "Highway",
    # Local-class (arterials / through streets)
    "secondary": "Local", "tertiary": "Local",
    "secondary_link": "Local", "tertiary_link": "Local",
    "unclassified": "Local",
    # Internal-class (residential / service / small)
    "residential": "Internal", "service": "Internal", "living_street": "Internal",
    "road": "Internal", "track": "Internal",
}


def fetch_road_type_osm(lat: float, lng: float, radius_m: int = 60) -> str | None:
    """Query Overpass for the nearest `highway=*` way and map it to our
    Road_Type taxonomy. Returns None on any failure."""
    query = (
        f"[out:json][timeout:10];"
        f"way(around:{radius_m},{lat},{lng})[highway];"
        f"out tags 5;"
    )
    mirrors = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    for url in mirrors:
        try:
            data = _http_post_json(url, "data=" + urllib.parse.quote(query))
            elements = data.get("elements") or []
            if not elements:
                continue
            # Pick the most "important" highway tag among hits.
            order = ["motorway", "trunk", "primary", "secondary", "tertiary",
                     "unclassified", "residential", "service", "living_street",
                     "road", "track"]
            best = None
            for el in elements:
                hw = (el.get("tags") or {}).get("highway", "").replace("_link", "")
                if hw in order and (best is None or order.index(hw) < order.index(best)):
                    best = hw
            if best:
                return _OSM_HIGHWAY_TO_TYPE.get(best, "Local")
        except Exception as e:
            print(f"OSM fetch failed at {url}: {e}")
    return None


# Open-Meteo WMO weather code → our 5-class label
def _weather_code_to_label(code: int) -> str:
    if code == 0:
        return "Clear"
    if code in (1, 2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        return "Rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    return "Clear"


def fetch_weather_open_meteo(lat: float, lng: float) -> dict[str, Any] | None:
    """Fetch current weather and return values already in IMPERIAL units
    (matching the model's training distribution)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lng}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,"
        "visibility,weather_code,wind_speed_10m,surface_pressure"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph"
        "&precipitation_unit=inch"
    )
    try:
        data = _http_get_json(url, timeout=8.0)
        cur = data.get("current") or {}
        if not cur:
            return None
        # visibility comes back in meters; convert to miles.
        vis_m = cur.get("visibility")
        vis_mi = (vis_m / 1609.344) if isinstance(vis_m, (int, float)) else None
        # surface_pressure is hPa; convert to inHg.
        pres_hpa = cur.get("surface_pressure")
        pres_in = (pres_hpa * 0.02953) if isinstance(pres_hpa, (int, float)) else None
        return {
            "temperature": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "wind_speed": cur.get("wind_speed_10m"),
            "precipitation": cur.get("precipitation"),
            "visibility": vis_mi,
            "pressure": pres_in,
            "weather_condition": _weather_code_to_label(int(cur.get("weather_code") or 0)),
        }
    except Exception as e:
        print(f"Open-Meteo fetch failed: {e}")
        return None


# ===========================================================================
# App
# ===========================================================================
app = FastAPI(title="TrafficIQ — Traffic & Severity Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Schemas
# ===========================================================================
class PredictionInput(BaseModel):
    latitude: float
    longitude: float
    temperature: float = Field(70.0)
    humidity: float = Field(50.0)
    pressure: float = Field(29.9)
    wind_speed: float = Field(5.0)
    visibility: float = Field(10.0)
    precipitation: float = Field(0.0)
    hour: int = Field(12, ge=0, le=23)
    weather_condition: str = Field("Clear")
    road_type: str = Field("Local")
    month: int = Field(default_factory=lambda: datetime.now().month, ge=1, le=12)
    units: str = Field(
        "imperial",
        description="'imperial' (°F, mph, in, mi, inHg) or "
        "'metric' (°C, km/h, mm, km, hPa). "
        "Metric inputs are auto-converted server-side.",
    )
    auto_fetch: bool = Field(
        True,
        description="If true, server pulls road_type from OSM (Overpass) and "
        "weather fields from Open-Meteo for the given lat/lng, overriding the "
        "supplied values. Set false to use only client-supplied inputs.",
    )


class RouteInput(PredictionInput):
    # We reuse all PredictionInput fields and add the route-specific ones.
    latitude: float = 0.0
    longitude: float = 0.0
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    segments: int = Field(15, ge=2, le=60)


# ===========================================================================
# Model loading (auto-discovery from models folder)
# ===========================================================================
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Store all discovered models
LOADED_MODELS = {}


def _try_load(label: str, *names: str):
    """Try a list of candidate filenames in MODELS_DIR; return the first one
    that loads, or None."""
    for name in names:
        p = MODELS_DIR / name
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                print(f"API: failed to load {label} from {name}: {e}")
    return None


def _auto_load_models():
    """Dynamically load all .pkl model files from MODELS_DIR."""
    if not MODELS_DIR.exists():
        print(f"API: Models directory not found at {MODELS_DIR}")
        return

    pkl_files = sorted(MODELS_DIR.glob("*.pkl"))
    print(f"\nAPI: Found {len(pkl_files)} model files in {MODELS_DIR.name}/")

    for pkl_file in pkl_files:
        model_name = pkl_file.stem
        try:
            model = joblib.load(pkl_file)
            LOADED_MODELS[model_name] = model
            print(f"  ✓ {model_name}")
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")


# Auto-discover and load all models
_auto_load_models()

# Legacy explicit variables for backward compatibility
traffic_model = LOADED_MODELS.get("traffic_model")
scaler_traffic = LOADED_MODELS.get("scaler_traffic")
traffic_features = LOADED_MODELS.get("traffic_features", [])
TRAFFIC_LOADED = traffic_model is not None and scaler_traffic is not None
if traffic_features:
    traffic_features = list(traffic_features)
if TRAFFIC_LOADED:
    print(f"API: Traffic model loaded ({len(traffic_features)} features).")

severity_model = LOADED_MODELS.get("severity_model")
scaler_severity = LOADED_MODELS.get("scaler_severity")
severity_features = LOADED_MODELS.get("severity_features", [])
SEVERITY_LOADED = severity_model is not None and scaler_severity is not None
if severity_features:
    severity_features = list(severity_features)
if SEVERITY_LOADED:
    print(f"API: Severity model loaded ({len(severity_features)} features).")

multi_model = LOADED_MODELS.get("xgboosst")
MULTI_LOADED = multi_model is not None
if MULTI_LOADED:
    print("API: Legacy multi-output model loaded as fallback.")
MULTI_FEATURES = [
    "Start_Lat",
    "Start_Lng",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Hour",
    "Month",
    "Road_Width(m)",
    "Speed_Limit(mph)",
    "Is_RushHour",
    "Is_Night",
    "Weather_Enc",
    "Road_Type_Enc",
]

# ---------------------------------------------------------------------------
# Spatiotemporal traffic model (empirical density, leakage-free)
# ---------------------------------------------------------------------------
ST_MODEL = LOADED_MODELS.get("traffic_st_model")
ST_FEATURES = LOADED_MODELS.get("traffic_st_features", [])
ST_CELL_LOOKUP = None
ST_CONFIG: dict = {}
ST_LOADED = False

if ST_MODEL is not None and ST_FEATURES:
    try:
        ST_FEATURES = list(ST_FEATURES)
        DATA_DIR = MODELS_DIR.parent / "data" / "processed"
        lookup_path = DATA_DIR / "cell_baseline_lookup.csv"
        config_path = DATA_DIR / "spatiotemporal_config.json"
        if lookup_path.exists() and config_path.exists():
            ST_CELL_LOOKUP = pd.read_csv(lookup_path).set_index("Cell_ID")
            with open(config_path) as fh:
                ST_CONFIG = json.load(fh)
            ST_LOADED = True
            print(
                f"API: Spatiotemporal model loaded "
                f"(cell_size={ST_CONFIG.get('cell_size_deg')}°, "
                f"{len(ST_CELL_LOOKUP):,} known cells, "
                f"{len(ST_FEATURES)} features)."
            )
        else:
            print(
                "API: Spatiotemporal model present but lookup/config missing; "
                "skipping. Re-run spark/spatiotemporal_labeling.py."
            )
    except Exception as e:
        print(f"API: spatiotemporal model unavailable. {e}")

# ---------------------------------------------------------------------------
# Geo-agnostic models (drop-in replacements for use outside the US)
# ---------------------------------------------------------------------------
SEV_GA_MODEL = LOADED_MODELS.get("severity_model_geo_agnostic")
SEV_GA_SCALER = LOADED_MODELS.get("scaler_severity_geo_agnostic")
SEV_GA_FEATURES = LOADED_MODELS.get("severity_features_geo_agnostic", [])
SEV_GA_LOADED = False
if SEV_GA_MODEL is not None and SEV_GA_SCALER is not None and SEV_GA_FEATURES:
    SEV_GA_FEATURES = list(SEV_GA_FEATURES)
    SEV_GA_LOADED = True
    print(f"API: Geo-agnostic severity model loaded ({len(SEV_GA_FEATURES)} features).")

ST_GA_MODEL = LOADED_MODELS.get("traffic_st_model_geo_agnostic")
ST_GA_FEATURES = LOADED_MODELS.get("traffic_st_features_geo_agnostic", [])
ST_GA_LOADED = False
if ST_GA_MODEL is not None and ST_GA_FEATURES:
    ST_GA_FEATURES = list(ST_GA_FEATURES)
    ST_GA_LOADED = True
    print(
        f"API: Geo-agnostic spatiotemporal model loaded ({len(ST_GA_FEATURES)} features)."
    )

print(f"\nAPI: Total {len(LOADED_MODELS)} model file(s) in memory.\n")


# ===========================================================================
# Domain knowledge
# ===========================================================================
ROAD_SPECS = {
    "Highway": {"enc": 3, "width": 34.5, "speed": 75.0},
    "Local": {"enc": 1, "width": 22.5, "speed": 45.0},
    "Internal": {"enc": 2, "width": 18.5, "speed": 25.0},
}
WEATHER_ENC = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Snow": 3, "Fog": 4}


def classify(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


def spatial_junction_signal(lat: float, lng: float) -> tuple[int, int]:
    """Deterministic stand-in for unknown Junction/Traffic_Signal flags."""
    return (
        1 if abs(lat * 1000) % 1 < 0.22 else 0,
        1 if abs(lng * 1000) % 1 < 0.30 else 0,
    )


def build_features(
    inp: dict[str, Any],
    lat: float,
    lng: float,
    road: dict[str, Any],
    weather_enc: int,
    junction: int,
    signal: int,
) -> dict[str, Any]:
    hour = inp["hour"]
    return {
        "Start_Lat": lat,
        "Start_Lng": lng,
        "Temperature(F)": inp.get("temperature", 70.0),
        "Humidity(%)": inp.get("humidity", 50.0),
        "Pressure(in)": inp.get("pressure", 29.9),
        "Visibility(mi)": inp.get("visibility", 10.0),
        "Wind_Speed(mph)": inp.get("wind_speed", 5.0),
        "Precipitation(in)": inp.get("precipitation", 0.0),
        "Weather_Enc": weather_enc,
        "Hour": hour,
        "Month": inp.get("month", datetime.now().month),
        "Is_RushHour": 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0,
        "Is_Night": 1 if (hour >= 21 or hour <= 5) else 0,
        "Road_Type_Enc": road["enc"],
        "Road_Width(m)": road["width"],
        "Speed_Limit(mph)": road["speed"],
        "Junction": junction,
        "Traffic_Signal": signal,
        "Amenity": 0,
        "Bump": 0,
        "Crossing": 0,
        "Give_Way": 0,
        "No_Exit": 0,
        "Railway": 0,
        "Roundabout": 0,
        "Station": 0,
        "Stop": 0,
        "Traffic_Calming": 0,
    }


# ===========================================================================
# Predictors
# ===========================================================================
def _to_frame(feat: dict[str, Any], cols: list[str]) -> pd.DataFrame:
    """Build a 1-row DataFrame in the EXACT column order the model expects.
    Using a DataFrame (not a numpy array) preserves feature names and silences
    the `X does not have valid feature names` sklearn warning."""
    return pd.DataFrame([{c: feat[c] for c in cols}], columns=cols)


def _predict_traffic(feat: dict[str, Any]) -> float:
    df = _to_frame(feat, traffic_features)
    scaled = scaler_traffic.transform(df)  # type: ignore[union-attr]
    return float(traffic_model.predict_proba(scaled)[0][1])  # type: ignore[union-attr]


def _predict_severity(feat: dict[str, Any]) -> float:
    df = _to_frame(feat, severity_features)
    scaled = scaler_severity.transform(df)  # type: ignore[union-attr]
    return float(severity_model.predict_proba(scaled)[0][1])  # type: ignore[union-attr]


def _predict_multi(feat: dict[str, Any]) -> tuple[float, float]:
    """Returns (traffic_prob, severity_prob) from the legacy multi-output model."""
    df = _to_frame(feat, MULTI_FEATURES)
    estimators = multi_model.estimators_  # type: ignore[union-attr]
    return (
        float(estimators[0].predict_proba(df)[0][1]),
        float(estimators[1].predict_proba(df)[0][1]),
    )


def _predict_severity_geo_agnostic(feat: dict[str, Any]) -> float:
    """Predict severity probability without using lat/lng."""
    df = _to_frame(feat, SEV_GA_FEATURES)
    scaled = SEV_GA_SCALER.transform(df)  # type: ignore[union-attr]
    return float(SEV_GA_MODEL.predict_proba(scaled)[0][1])  # type: ignore[union-attr]


def _predict_st_geo_agnostic(
    inp: dict[str, Any],
    road: dict[str, Any],
    weather_enc: int,
    junction: int,
    signal: int,
) -> dict[str, Any]:
    """3-class spatiotemporal prediction without any geographic features."""
    feat = {
        "Hour": inp["hour"],
        "mean_Month": float(inp.get("month", datetime.now().month)),
        "mean_Temperature_F_": inp.get("temperature", 70.0),
        "mean_Humidity_Pct_": inp.get("humidity", 50.0),
        "mean_Pressure_in_": inp.get("pressure", 29.9),
        "mean_Visibility_mi_": inp.get("visibility", 10.0),
        "mean_Wind_Speed_mph_": inp.get("wind_speed", 5.0),
        "mean_Precipitation_in_": inp.get("precipitation", 0.0),
        "mean_Weather_Enc": float(weather_enc),
        "mean_Road_Type_Enc": float(road["enc"]),
        "frac_Junction": float(junction),
        "frac_Traffic_Signal": float(signal),
        "frac_Crossing": 0.0,
        "frac_Stop": 0.0,
        "frac_Roundabout": 0.0,
        "frac_Station": 0.0,
        "frac_Amenity": 0.0,
    }
    X = pd.DataFrame([{c: feat[c] for c in ST_GA_FEATURES}], columns=ST_GA_FEATURES)
    probs = ST_GA_MODEL.predict_proba(X)[0]  # type: ignore[union-attr]
    pred = int(np.argmax(probs))
    return {
        "traffic_level": ["Low", "Medium", "High"][pred],
        "class_index": pred,
        "probabilities": {
            "Low": round(float(probs[0]), 4),
            "Medium": round(float(probs[1]), 4),
            "High": round(float(probs[2]), 4),
        },
    }


# ---------------------------------------------------------------------------
# Spatiotemporal helpers
# ---------------------------------------------------------------------------
def _cell_id_for(lat: float, lng: float, cell_size: float) -> str:
    """Reproduce the labeler's cell quantization at inference time."""
    return f"{round(lat / cell_size)}_{round(lng / cell_size)}"


def _st_baselines(lat: float, lng: float) -> dict[str, float]:
    """Look up cell baselines for a clicked point, falling back to global
    medians when the cell is unseen."""
    if ST_CELL_LOOKUP is None:
        # No lookup available — return neutral baselines.
        return {
            "cell_total_accidents": 5.0,
            "cell_unique_hours": 5.0,
            "cell_unique_months": 3.0,
            "Cell_Lat_Mean": lat,
            "Cell_Lng_Mean": lng,
        }

    cell_size = ST_CONFIG.get("cell_size_deg", 0.05)
    cid = _cell_id_for(lat, lng, cell_size)

    if cid in ST_CELL_LOOKUP.index:
        row = ST_CELL_LOOKUP.loc[[cid]].iloc[0]

        return {
            "cell_total_accidents": float(row["cell_total_accidents"]),
            "cell_unique_hours": float(row["cell_unique_hours"]),
            "cell_unique_months": float(row["cell_unique_months"]),
            "Cell_Lat_Mean": float(row["Cell_Lat_Mean"]),
            "Cell_Lng_Mean": float(row["Cell_Lng_Mean"]),
        }

    # Cold-start: use medians of the whole lookup as a generic prior.
    return {
        "cell_total_accidents": float(ST_CELL_LOOKUP["cell_total_accidents"].median()),
        "cell_unique_hours": float(ST_CELL_LOOKUP["cell_unique_hours"].median()),
        "cell_unique_months": float(ST_CELL_LOOKUP["cell_unique_months"].median()),
        "Cell_Lat_Mean": float(lat),
        "Cell_Lng_Mean": float(lng),
    }


def _predict_spatiotemporal(
    inp: dict[str, Any],
    lat: float,
    lng: float,
    road: dict[str, Any],
    weather_enc: int,
    junction: int,
    signal: int,
) -> dict[str, Any]:
    """Predict (Low/Medium/High) traffic level via the spatiotemporal model."""
    if not ST_LOADED:
        raise RuntimeError("Spatiotemporal model is not loaded.")

    base = _st_baselines(lat, lng)

    feat = {
        # spatial
        "Cell_Lat_Mean": base["Cell_Lat_Mean"],
        "Cell_Lng_Mean": base["Cell_Lng_Mean"],
        # temporal
        "Hour": inp["hour"],
        "mean_Month": float(inp.get("month", datetime.now().month)),
        # weather (point-in-time conditions; the model expects "mean_*" names)
        "mean_Temperature_F_": inp.get("temperature", 70.0),
        "mean_Humidity_Pct_": inp.get("humidity", 50.0),
        "mean_Pressure_in_": inp.get("pressure", 29.9),
        "mean_Visibility_mi_": inp.get("visibility", 10.0),
        "mean_Wind_Speed_mph_": inp.get("wind_speed", 5.0),
        "mean_Precipitation_in_": inp.get("precipitation", 0.0),
        "mean_Weather_Enc": float(weather_enc),
        # road / infra
        "mean_Road_Type_Enc": float(road["enc"]),
        "frac_Junction": float(junction),
        "frac_Traffic_Signal": float(signal),
        "frac_Crossing": 0.0,
        "frac_Stop": 0.0,
        "frac_Roundabout": 0.0,
        "frac_Station": 0.0,
        "frac_Amenity": 0.0,
        # baselines
        "cell_total_accidents": base["cell_total_accidents"],
        "cell_unique_hours": base["cell_unique_hours"],
        "cell_unique_months": base["cell_unique_months"],
    }

    # Build a 1-row DataFrame in the exact column order the model was trained on.
    X = pd.DataFrame([{c: feat[c] for c in ST_FEATURES}], columns=ST_FEATURES)
    if ST_MODEL is None:
        raise ValueError("ST_MODEL is not loaded")
    probs = ST_MODEL.predict_proba(X)[0]
    pred_class = int(np.argmax(probs))
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    return {
        "traffic_level": label_map[pred_class],
        "class_index": pred_class,
        "probabilities": {
            "Low": round(float(probs[0]), 4),
            "Medium": round(float(probs[1]), 4),
            "High": round(float(probs[2]), 4),
        },
        "cell_id": _cell_id_for(lat, lng, ST_CONFIG.get("cell_size_deg", 0.05)),
        "cell_known": _cell_id_for(lat, lng, ST_CONFIG.get("cell_size_deg", 0.05))
        in (ST_CELL_LOOKUP.index if ST_CELL_LOOKUP is not None else []),
    }


def simulate_traffic(
    lat: float, lng: float, hour: int, weather: str, road_key: str, visibility: float
) -> tuple[float, dict]:
    road = ROAD_SPECS.get(road_key, ROAD_SPECS["Local"])
    is_rush = (6 <= hour <= 9) or (16 <= hour <= 19)
    p = 0.30
    if is_rush:
        p += 0.30
    if weather in ("Rain", "Snow"):
        p += 0.18
    if weather == "Fog":
        p += 0.12
    if road["enc"] == 2:
        p += 0.10
    if visibility < 3:
        p += 0.08
    h = (abs(np.sin(lat * 12.9898 + lng * 78.233) * 43758.5453)) % 1
    p = max(0.05, min(0.95, p + (h - 0.5) * 0.4))
    return p, road


def simulate_severity(
    lat: float, lng: float, hour: int, weather: str, visibility: float
) -> float:
    p = 0.20
    if weather in ("Rain", "Snow", "Fog"):
        p += 0.18
    if visibility < 3:
        p += 0.10
    if hour >= 21 or hour <= 5:
        p += 0.10
    h = (abs(np.cos(lat * 51.31 + lng * 17.7) * 12345.6789)) % 1
    return max(0.02, min(0.95, p + (h - 0.5) * 0.3))


def predict_pair(feats: dict[str, Any], inp: dict[str, Any]) -> tuple[float, float]:
    """Run both models with graceful fallback through:
    per-target → legacy multi-output → deterministic simulator."""
    t_prob: float | None = None
    s_prob: float | None = None

    if TRAFFIC_LOADED:
        try:
            t_prob = _predict_traffic(feats)
        except Exception as e:
            print(f"traffic predict error: {e}")

    if SEVERITY_LOADED:
        try:
            s_prob = _predict_severity(feats)
        except Exception as e:
            print(f"severity predict error: {e}")

    if (t_prob is None or s_prob is None) and MULTI_LOADED:
        try:
            mt, ms = _predict_multi(feats)
            if t_prob is None:
                t_prob = mt
            if s_prob is None:
                s_prob = ms
        except Exception as e:
            print(f"multi predict error: {e}")

    if t_prob is None:
        t_prob, _ = simulate_traffic(
            inp["latitude"],
            inp["longitude"],
            inp["hour"],
            inp["weather_condition"],
            inp["road_type"],
            inp["visibility"],
        )
    if s_prob is None:
        s_prob = simulate_severity(
            inp["latitude"],
            inp["longitude"],
            inp["hour"],
            inp["weather_condition"],
            inp["visibility"],
        )
    return t_prob, s_prob


# ===========================================================================
# Endpoints
# ===========================================================================
@app.get("/")
def health():
    return {
        "status": "healthy",
        "traffic_loaded": TRAFFIC_LOADED,
        "severity_loaded": SEVERITY_LOADED,
        "models_loaded": (
            TRAFFIC_LOADED
            or SEVERITY_LOADED
            or MULTI_LOADED
            or ST_LOADED
            or SEV_GA_LOADED
            or ST_GA_LOADED
        ),
        "geo_aware": {
            "traffic": TRAFFIC_LOADED,
            "severity": SEVERITY_LOADED,
            "multi": MULTI_LOADED,
            "spatiotemporal": ST_LOADED,
            "spatiotemporal_cells": (
                len(ST_CELL_LOOKUP) if ST_CELL_LOOKUP is not None else 0
            ),
        },
        "geo_agnostic": {
            "severity": SEV_GA_LOADED,
            "spatiotemporal": ST_GA_LOADED,
        },
        "us_bbox": US_BBOX,
    }


@app.get("/weather")
def get_weather(lat: float, lng: float):
    """Server-side proxy to Open-Meteo."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lng}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,"
        "visibility,weather_code,wind_speed_10m"
    )
    try:
        return _http_get_json(url, timeout=8.0)
    except Exception as e:
        return {"error": str(e)}


@app.get("/route")
def get_route(start_lat: float, start_lng: float, end_lat: float, end_lng: float):
    """Server-side proxy to OSRM with one fallback mirror."""
    mirrors = [
        "https://router.project-osrm.org",
        "https://routing.openstreetmap.de/routed-car",
    ]
    path = (
        f"/route/v1/driving/{start_lng},{start_lat};{end_lng},{end_lat}"
        "?overview=full&geometries=geojson&steps=true&alternatives=true"
    )
    last_err: str | None = None
    for base in mirrors:
        try:
            data = _http_get_json(base + path, timeout=6.0)
            if data and data.get("code") == "Ok":
                return data
            last_err = f"bad response from {base}"
        except Exception as e:
            last_err = str(e)
    return {"code": "Error", "error": last_err or "all mirrors failed"}


@app.post("/predict")
def predict(inp: PredictionInput):
    # 1) Convert metric → imperial if requested
    payload = normalize_units(inp.model_dump())

    # 1b) Auto-fetch live road type (OSM) + weather (Open-Meteo) when requested.
    sources: dict[str, str] = {"road_type": "client", "weather": "client"}
    if payload.get("auto_fetch", True):
        rt = fetch_road_type_osm(payload["latitude"], payload["longitude"])
        if rt:
            payload["road_type"] = rt
            sources["road_type"] = "osm"
        wx = fetch_weather_open_meteo(payload["latitude"], payload["longitude"])
        if wx:
            for k in ("temperature", "humidity", "wind_speed", "precipitation",
                      "visibility", "pressure", "weather_condition"):
                if wx.get(k) is not None:
                    payload[k] = wx[k]
            sources["weather"] = "open-meteo"

    # 2) Detect region — inside US uses geo-aware models, outside uses geo-agnostic
    in_us = is_in_continental_us(payload["latitude"], payload["longitude"])
    region = "us" if in_us else "outside_us"

    # 3) Build the standard feature dict (works for both branches)
    road = ROAD_SPECS.get(payload["road_type"], ROAD_SPECS["Local"])
    weather_enc = WEATHER_ENC.get(payload["weather_condition"], 0)
    j, s = spatial_junction_signal(payload["latitude"], payload["longitude"])
    feats = build_features(
        payload, payload["latitude"], payload["longitude"], road, weather_enc, j, s
    )
    is_rush = 1 if (6 <= payload["hour"] <= 9) or (16 <= payload["hour"] <= 19) else 0

    # 4) Route
    response: dict[str, Any] = {
        "region": region,
        "models_used": {},
        "details": {
            "road_type": payload["road_type"],
            "road_width": road["width"],
            "speed_limit": road["speed"],
            "is_rush_hour": bool(is_rush),
            "units_received": str(payload.get("units", "imperial")),
            "weather_condition": payload["weather_condition"],
            "temperature": payload["temperature"],
            "humidity": payload["humidity"],
            "wind_speed": payload["wind_speed"],
            "visibility": payload["visibility"],
            "precipitation": payload["precipitation"],
            "pressure": payload["pressure"],
        },
        "sources": sources,
    }

    if in_us:
        # ---- US: geo-aware models (existing pipeline) ----
        t_prob, s_prob = predict_pair(feats, payload)
        response.update(
            {
                "traffic_probability": round(t_prob, 4),
                "traffic_level": classify(t_prob),
                "accident_probability": round(s_prob, 4),
                "risk_level": classify(s_prob),
            }
        )
        # Determine if real models were used or if it fell back to simulation
        response["models_used"]["traffic"] = (
            "geo_aware" if TRAFFIC_LOADED else "simulator"
        )
        response["models_used"]["severity"] = (
            "geo_aware" if SEVERITY_LOADED else "simulator"
        )

        if ST_LOADED:
            try:
                response["spatiotemporal"] = _predict_spatiotemporal(
                    payload,
                    payload["latitude"],
                    payload["longitude"],
                    road,
                    weather_enc,
                    j,
                    s,
                )
                response["spatiotemporal"]["model"] = "geo_aware"
            except Exception as e:
                response["spatiotemporal_error"] = str(e)
    else:
        # ---- Outside US: geo-agnostic models ----
        # Severity: use geo-agnostic if loaded, else gracefully degrade
        if SEV_GA_LOADED:
            try:
                s_prob = _predict_severity_geo_agnostic(feats)
                response["accident_probability"] = round(s_prob, 4)
                response["risk_level"] = classify(s_prob)
                response["models_used"]["severity"] = "geo_agnostic"
            except Exception as e:
                response["severity_error"] = str(e)
        else:
            s_prob = simulate_severity(
                payload["latitude"],
                payload["longitude"],
                payload["hour"],
                payload["weather_condition"],
                payload["visibility"],
            )
            response["accident_probability"] = round(s_prob, 4)
            response["risk_level"] = classify(s_prob)
            response["models_used"]["severity"] = "simulator"

        # Traffic: geo-agnostic spatiotemporal IS the traffic prediction
        if ST_GA_LOADED:
            try:
                st_out = _predict_st_geo_agnostic(payload, road, weather_enc, j, s)
                st_out["model"] = "geo_agnostic"
                response["spatiotemporal"] = st_out
                # Bridge to legacy fields so existing dashboards keep working:
                # map class_index 0/1/2 → probability 0.2/0.5/0.8 for the bar.
                bridge_prob = (0.20, 0.55, 0.85)[st_out["class_index"]]
                response["traffic_probability"] = bridge_prob
                response["traffic_level"] = st_out["traffic_level"]
                response["models_used"]["traffic"] = "geo_agnostic_spatiotemporal"
            except Exception as e:
                response["spatiotemporal_error"] = str(e)
        else:
            # Fallback for traffic when geo-agnostic models aren't trained
            t_prob, _ = simulate_traffic(
                payload["latitude"],
                payload["longitude"],
                payload["hour"],
                payload["weather_condition"],
                payload["road_type"],
                payload["visibility"],
            )
            response["traffic_probability"] = round(t_prob, 4)
            response["traffic_level"] = classify(t_prob)
            response["models_used"]["traffic"] = "simulator"

        response["note"] = (
            "Models trained on US Accidents 2016-2023. Outside the US the "
            "geo-agnostic variant is used: it relies only on hour, weather, "
            "road type and infrastructure — features that transfer between "
            "countries. Spatial accident-density priors are not available for "
            "this region; predictions reflect generic time-and-conditions "
            "patterns, not local Cairo traffic dynamics."
        )

    return response


@app.post("/predict-spatiotemporal")
def predict_spatiotemporal(inp: PredictionInput):
    """Dedicated endpoint that returns ONLY the spatiotemporal traffic
    prediction — useful for clients that don't need the binary congestion
    or severity outputs."""
    if not ST_LOADED:
        return {
            "error": "Spatiotemporal model not loaded. "
            "Run spark/spatiotemporal_labeling.py followed by "
            "models/train_traffic_spatiotemporal.py."
        }
    road = ROAD_SPECS.get(inp.road_type, ROAD_SPECS["Local"])
    weather_enc = WEATHER_ENC.get(inp.weather_condition, 0)
    j, s = spatial_junction_signal(inp.latitude, inp.longitude)
    return _predict_spatiotemporal(
        inp.model_dump(),
        inp.latitude,
        inp.longitude,
        road,
        weather_enc,
        j,
        s,
    )


@app.post("/predict-route")
def predict_route(inp: RouteInput):
    n = inp.segments
    base = inp.model_dump()

    # Fetch a single weather sample at the route midpoint (the route is short
    # enough that current conditions don't vary meaningfully across it),
    # but resolve OSM road type per-point so we follow real road class.
    if base.get("auto_fetch", True):
        mid_lat = (inp.start_lat + inp.end_lat) / 2
        mid_lng = (inp.start_lng + inp.end_lng) / 2
        wx = fetch_weather_open_meteo(mid_lat, mid_lng)
        if wx:
            for k in ("temperature", "humidity", "wind_speed", "precipitation",
                      "visibility", "pressure", "weather_condition"):
                if wx.get(k) is not None:
                    base[k] = wx[k]

    weather_enc = WEATHER_ENC.get(base["weather_condition"], 0)

    points = [
        (
            inp.start_lat + (i / n) * (inp.end_lat - inp.start_lat),
            inp.start_lng + (i / n) * (inp.end_lng - inp.start_lng),
        )
        for i in range(n + 1)
    ]

    t_probs, s_probs, road_types = [], [], []
    default_road = ROAD_SPECS.get(inp.road_type, ROAD_SPECS["Local"])
    for lat, lng in points:
        if base.get("auto_fetch", True):
            rt = fetch_road_type_osm(lat, lng) or inp.road_type
        else:
            rt = inp.road_type
        road = ROAD_SPECS.get(rt, default_road)
        road_types.append(rt)
        j, sg = spatial_junction_signal(lat, lng)
        feats = build_features(base, lat, lng, road, weather_enc, j, sg)
        per_pt = {**base, "latitude": lat, "longitude": lng}
        t, s = predict_pair(feats, per_pt)
        t_probs.append(t)
        s_probs.append(s)

    segments = []
    for i in range(n):
        tp = (t_probs[i] + t_probs[i + 1]) / 2
        sp = (s_probs[i] + s_probs[i + 1]) / 2
        segments.append(
            {
                "start": {"lat": points[i][0], "lng": points[i][1]},
                "end": {"lat": points[i + 1][0], "lng": points[i + 1][1]},
                "road_type": road_types[i],
                "traffic_probability": round(tp, 4),
                "traffic_level": classify(tp),
                "accident_probability": round(sp, 4),
                "risk_level": classify(sp),
            }
        )

    return {
        "segments": segments,
        "road_type": inp.road_type,
        "hour": inp.hour,
        "models_used": TRAFFIC_LOADED or SEVERITY_LOADED or MULTI_LOADED,
    }


if __name__ == "__main__":
    print("API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
