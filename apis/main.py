from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import uvicorn

# ==========================================
# APP INITIALIZATION
# ==========================================
app = FastAPI(title="Road Real-Time Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PYDANTIC MODELS (Input Schema)
# ==========================================
class PredictionInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    temperature: float = Field(70.0, description="Temperature in F")
    humidity: float = Field(50.0, description="Humidity in %")
    pressure: float = Field(29.9, description="Pressure in inches")
    visibility: float = Field(10.0, description="Visibility in miles")
    wind_speed: float = Field(5.0, description="Wind speed in mph")
    precipitation: float = Field(0.0, description="Precipitation in inches")
    hour: int = Field(12, ge=0, le=23, description="Hour of the day")
    weather_condition: str = Field("Clear", description="Weather condition")
    road_type: str = Field("Local", description="Road type (Highway, Local, Internal)")


# ==========================================
# LOAD ML MODELS & ARTIFACTS SAFELY
# ==========================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

severity_model = None
traffic_model = None
scaler_severity = None
scaler_traffic = None
severity_features = []
traffic_features = []
MODELS_LOADED = False


try:
    # Define types so VS Code stops complaining
    severity_model: RandomForestClassifier = joblib.load(
        os.path.join(MODELS_DIR, "severity_model.pkl")
    )
    traffic_model: RandomForestClassifier = joblib.load(
        os.path.join(MODELS_DIR, "traffic_model.pkl")
    )
    scaler_severity: StandardScaler = joblib.load(
        os.path.join(MODELS_DIR, "scaler_severity.pkl")
    )
    scaler_traffic: StandardScaler = joblib.load(
        os.path.join(MODELS_DIR, "scaler_traffic.pkl")
    )
    severity_features: list = joblib.load(
        os.path.join(MODELS_DIR, "severity_features.pkl")
    )
    traffic_features: list = joblib.load(
        os.path.join(MODELS_DIR, "traffic_features.pkl")
    )
    MODELS_LOADED = True
    print("API: ML Models loaded successfully.")
except Exception as e:
    print(f"API Warning: Could not load models. Running in Demo Mode. Error: {e}")


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_road_specs(road_type_str):
    specs = {
        "Highway": {"enc": 0, "width": 34.0, "speed": 70.0},
        "Local": {"enc": 1, "width": 22.5, "speed": 40.0},
        "Internal": {"enc": 2, "width": 18.5, "speed": 25.0},
    }
    return specs.get(road_type_str, specs["Local"])


def map_weather(weather_str):
    weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Snow": 3, "Fog": 4}
    return weather_map.get(weather_str, 0)


def simulate_logic(input_data):
    """Fallback logic if models fail to load or predict"""
    road = get_road_specs(input_data.road_type)
    is_rush = 1 if (6 <= input_data.hour <= 9) or (16 <= input_data.hour <= 19) else 0

    prob = 0.2
    if is_rush:
        prob += 0.2
    if input_data.weather_condition in ["Rain", "Snow", "Fog"]:
        prob += 0.15
    if input_data.visibility < 5:
        prob += 0.1

    traf_prob = 0.3
    if is_rush:
        traf_prob += 0.4
    if input_data.weather_condition in ["Rain", "Snow"]:
        traf_prob += 0.15
    if road["enc"] == 2:
        traf_prob += 0.1

    return {
        "accident_probability": round(min(prob, 0.95), 4),
        "risk_level": "High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low"),
        "traffic_level": "High" if traf_prob >= 0.5 else "Low",
        "traffic_probability": round(min(traf_prob, 0.95), 4),
        "details": {
            "road_type": input_data.road_type,
            "road_width": road["width"],
            "speed_limit": road["speed"],
            "is_rush_hour": bool(is_rush),
        },
    }


# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {"status": "healthy", "models_loaded": MODELS_LOADED}


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # If models didn't load, use simulation instantly (prevents crashes)
        if not MODELS_LOADED:
            return simulate_logic(input_data)

        # 1. Prepare features
        road = get_road_specs(input_data.road_type)
        weather_enc = map_weather(input_data.weather_condition)
        is_rush = (
            1 if (6 <= input_data.hour <= 9) or (16 <= input_data.hour <= 19) else 0
        )
        is_night = 1 if (input_data.hour >= 21 or input_data.hour <= 5) else 0

        feat_dict = {
            "Start_Lat": input_data.latitude,
            "Start_Lng": input_data.longitude,
            "Temperature(F)": input_data.temperature,
            "Humidity(%)": input_data.humidity,
            "Pressure(in)": input_data.pressure,
            "Visibility(mi)": input_data.visibility,
            "Wind_Speed(mph)": input_data.wind_speed,
            "Precipitation(in)": input_data.precipitation,
            "Weather_Enc": weather_enc,
            "Hour": input_data.hour,
            "Month": 6,
            "Is_RushHour": is_rush,
            "Is_Night": is_night,
            "Road_Type_Enc": road["enc"],
            "Road_Width(m)": road["width"],
            "Speed_Limit(mph)": road["speed"],
        }

        for col in [
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
        ]:
            feat_dict[col] = 0

        # --- PREDICT ACCIDENT SEVERITY (Safe Extraction) ---
        severity_values = [feat_dict[col] for col in severity_features]
        severity_array = np.array(severity_values).reshape(1, -1)
        severity_scaled = scaler_severity.transform(severity_array)

        # THE FIX: Check if model has 2 classes before accessing index 1
        if len(severity_model.classes_) == 2:
            accident_prob = severity_model.predict_proba(severity_scaled)[0][1]
        else:
            accident_prob = 0.0  # Fallback if training data only had 1 class

        # --- PREDICT TRAFFIC CONGESTION (Safe Extraction) ---
        traffic_values = [feat_dict[col] for col in traffic_features]
        traffic_array = np.array(traffic_values).reshape(1, -1)
        traffic_scaled = scaler_traffic.transform(traffic_array)

        if len(traffic_model.classes_) == 2:
            traffic_prob = traffic_model.predict_proba(traffic_scaled)[0][1]
        else:
            traffic_prob = 0.0

        risk_level = (
            "High"
            if accident_prob >= 0.6
            else ("Medium" if accident_prob >= 0.3 else "Low")
        )
        traffic_level = "High" if traffic_prob >= 0.5 else "Low"

        return {
            "accident_probability": round(float(accident_prob), 4),
            "risk_level": risk_level,
            "traffic_level": traffic_level,
            "traffic_probability": round(float(traffic_prob), 4),
            "details": {
                "road_type": input_data.road_type,
                "road_width": road["width"],
                "speed_limit": road["speed"],
                "is_rush_hour": bool(is_rush),
            },
        }

    except Exception as e:
        # If ANY error happens during real prediction, fallback to simulation instead of crashing/hanging
        print(f"API Prediction Error: {e}. Falling back to simulation.")
        return simulate_logic(input_data)


if __name__ == "__main__":
    print("Starting API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
