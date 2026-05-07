import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="TrafficIQ Backend")

# Enable CORS for the dashboard to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Store loaded models
loaded_models: Dict[str, Any] = {}


# =========================
# Load Models
# =========================
def load_models():
    """Load all ML models from models directory."""

    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}")
        return

    for file in os.listdir(MODELS_DIR):

        if file.endswith((".joblib", ".pkl", ".xgb")):

            model_name = os.path.splitext(file)[0]
            model_path = os.path.join(MODELS_DIR, file)

            try:
                loaded_models[model_name] = joblib.load(model_path)
                print(f"Loaded model: {model_name}")

            except Exception as e:
                print(f"Failed to load {file}: {e}")


@app.on_event("startup")
async def startup_event():
    load_models()


# =========================
# Request Schema
# =========================
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    hour: int

    weather_condition: str
    road_type: str

    visibility: float
    precipitation: float
    temperature: float
    humidity: float
    wind_speed: float

    units: str = "imperial"


# =========================
# Root Endpoint
# =========================
@app.get("/")
async def root():

    has_traffic = any("traffic" in model.lower() for model in loaded_models.keys())

    has_severity = any(
        ("severity" in model.lower() or "accident" in model.lower())
        for model in loaded_models.keys()
    )

    return {
        "status": "online",
        "traffic_loaded": has_traffic,
        "severity_loaded": has_severity,
        "available_models": list(loaded_models.keys()),
    }


# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
async def predict(data: PredictionRequest):

    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded on server")

    # =========================
    # Unit Conversion
    # =========================
    temp = data.temperature
    vis = data.visibility
    precip = data.precipitation
    wind = data.wind_speed

    # Convert metric → imperial
    if data.units == "metric":

        temp = (temp * 9 / 5) + 32
        vis = vis / 1.60934
        precip = precip / 25.4
        wind = wind / 1.60934

    # =========================
    # Feature Engineering
    # =========================
    weather_map = {
        "clear": 0,
        "cloudy": 1,
        "clouds": 1,
        "overcast": 1,
        "rain": 2,
        "drizzle": 2,
        "mist": 2,
        "snow": 3,
        "fog": 4,
        "haze": 4,
    }

    road_map = {"Local": 1, "Internal": 2, "Highway": 3}

    weather_val = weather_map.get(data.weather_condition.lower(), 0)

    road_val = road_map.get(data.road_type, 1)

    # Determine road specifics for features
    road_width = 34.5 if data.road_type == "Highway" else 22.5
    speed_limit = 75.0 if data.road_type == "Highway" else 45.0
    is_rush = 1 if (6 <= data.hour <= 9) or (16 <= data.hour <= 19) else 0
    is_night = 1 if (data.hour >= 21 or data.hour <= 5) else 0
    month = 5  # Default month if not provided

    # Create a full feature dictionary containing all possible columns
    full_features = {
        "Start_Lat": data.latitude,
        "Start_Lng": data.longitude,
        "Hour": data.hour,
        "Month": month,
        "Is_RushHour": is_rush,
        "Is_Night": is_night,
        "Road_Type_Enc": road_val,
        "Road_Width(m)": road_width,
        "Speed_Limit(mph)": speed_limit,
        "Weather_Enc": weather_val,
        "Junction": 0,
        "Traffic_Signal": 0,
        "Visibility(mi)": vis,
        "Precipitation(in)": precip,
        "Temperature(F)": temp,
        "Humidity(%)": data.humidity,
        "Pressure(in)": 29.9,
        "Wind_Speed(mph)": wind,
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

    # IMPORTANT FIX
    response: Dict[str, Any] = {"details": {}}

    # =========================
    # Run Predictions
    # =========================
    for name, model in loaded_models.items():
        model_key = name.lower()

        # Skip if it's a scaler or feature list (just data artifacts)
        if "scaler" in model_key or "features" in model_key:
            continue

        try:
            # Determine which feature list and scaler to use
            prefix = "traffic" if "traffic" in model_key else "severity"
            feat_cols = loaded_models.get(f"{prefix}_features")
            scaler = loaded_models.get(f"scaler_{prefix}")

            if not feat_cols or not scaler:
                continue

            # Prepare DataFrame and scale
            df = pd.DataFrame(
                [{c: full_features[c] for c in feat_cols}], columns=feat_cols
            )
            scaled_df = scaler.transform(df)

            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(scaled_df)
                preds_arr = np.array(preds)

                if len(preds_arr.shape) > 1 and preds_arr.shape[1] > 1:
                    prob = float(preds_arr[0, 1])
                else:
                    prob = float(preds_arr[0])
            else:
                raw_pred = model.predict(scaled_df)
                prob = float(np.clip(raw_pred[0], 0, 1))

            # =========================
            # Traffic Model
            # =========================
            if "traffic" in model_key:

                response["traffic_probability"] = prob

                response["traffic_level"] = (
                    "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
                )

            # =========================
            # Accident / Severity Model
            # =========================
            elif "severity" in model_key or "accident" in model_key:

                response["accident_probability"] = prob

                response["risk_level"] = (
                    "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
                )

            # =========================
            # Generic Model Output
            # =========================
            else:

                response[f"{model_key}_output"] = prob

        except Exception as e:

            print(f"Prediction failed for model {name}: {e}")

    # =========================
    # Extra UI Metadata
    # =========================
    response["details"]["road_width"] = 34 if data.road_type == "Highway" else 22.5

    response["details"]["speed_limit"] = 70 if data.road_type == "Highway" else 40

    return response
