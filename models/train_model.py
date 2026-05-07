"""
Train the two TrafficIQ models on the labeled dataset.

Reads:  data/processed/final_dataset.csv
Writes (all under models/):
    traffic_model.pkl       — XGBoost binary classifier (Traffic_Congestion)
    scaler_traffic.pkl      — StandardScaler fit on the traffic features
    traffic_features.pkl    — list of feature names, in training order
    severity_model.pkl      — XGBoost binary classifier (High_Risk_Accident)
    scaler_severity.pkl     — StandardScaler fit on the severity features
    severity_features.pkl   — list of feature names, in training order

Run:
    python models/train_model.py

Notes
-----
1. **Why two SEPARATE XGBClassifiers instead of one MultiOutputClassifier?**
   The earlier `models/XGBoost.py` script wrapped XGBoost in
   `sklearn.multioutput.MultiOutputClassifier`. On modern scikit-learn that
   triggers:

       UserWarning: `sklearn.utils.parallel.delayed` should be used with
       `sklearn.utils.parallel.Parallel` ...

   The warning is harmless but noisy, and MultiOutput offers no real benefit
   for two independent binary targets. Training two XGBClassifiers directly
   is faster (uses XGBoost's native `n_jobs`), gives per-target threshold
   control, and has no compat warning.

2. **Labels are NOT recomputed here.** They are produced once in
   `spark/traffic_labeling.py` and trusted as ground truth. (The previous
   train_model.py recomputed `Traffic_Congestion` with a different formula,
   silently disagreeing with the labeling step.)

3. We pass DataFrames into `StandardScaler.fit/transform` AND into XGBoost.
   This preserves feature names so prediction-time code can pass DataFrames
   too, eliminating the warning:
       "X does not have valid feature names, but StandardScaler was fitted
        with feature names"
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Defensively silence the sklearn-parallel warning that some users see when
# using MultiOutputClassifier on older sklearn — even though we no longer use
# it, the warning can leak in from third-party imports.
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
MODELS_DIR = HERE  # models/

# ---------------------------------------------------------------------------
# Feature lists  — single source of truth for which columns each model sees
# ---------------------------------------------------------------------------
TRAFFIC_FEATURES = [
    "Start_Lat",
    "Start_Lng",
    "Hour",
    "Month",
    "Is_RushHour",
    "Is_Night",
    "Road_Type_Enc",
    "Road_Width(m)",
    "Speed_Limit(mph)",
    "Weather_Enc",
    "Junction",
    "Traffic_Signal",
    "Visibility(mi)",
    "Precipitation(in)",
]

SEVERITY_FEATURES = [
    "Start_Lat",
    "Start_Lng",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Enc",
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
    "Is_RushHour",
    "Is_Night",
    "Road_Type_Enc",
    "Road_Width(m)",
    "Speed_Limit(mph)",
]

XGB_KWARGS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
)


def _train_one(X: pd.DataFrame, y: pd.Series, name: str):
    """Train one XGBoost binary classifier with a StandardScaler in front."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Keep DataFrame structure so feature names propagate end-to-end.
    scaler: StandardScaler = StandardScaler()
    scaler.set_output(transform="pandas")   
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    pos = max(int((y_tr == 1).sum()), 1)
    neg = int((y_tr == 0).sum())
    pos_w = neg / pos
    print(f"   [{name}] scale_pos_weight={pos_w:.2f}  " f"(pos={pos:,}, neg={neg:,})")

    model = xgb.XGBClassifier(scale_pos_weight=pos_w, **XGB_KWARGS)
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_te_s)

    print(f"\n--- [{name}] Classification Report ---")
    print(classification_report(y_te, preds, digits=3))
    print(f"   Confusion matrix:\n{confusion_matrix(y_te, preds)}")
    print(f"   Accuracy: {accuracy_score(y_te, preds) * 100:.2f}%")
    return model, scaler


def train_models(
    input_path: os.PathLike | str = INPUT_PATH,
    models_dir: os.PathLike | str = MODELS_DIR,
) -> None:
    start = time.time()
    input_path = Path(input_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[TRAFFIC + SEVERITY — XGBoost] Starting...")
    print("=" * 60)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Final dataset not found at {input_path}. "
            "Run spark/data_cleaning.py -> feature_engineering.py -> "
            "traffic_labeling.py first."
        )

    df = pd.read_csv(input_path, low_memory=False).dropna()
    print(f"-> Records: {len(df):,}")

    # Sanity-check labels
    for col in ("Traffic_Congestion", "High_Risk_Accident"):
        if col not in df.columns:
            raise KeyError(f"`{col}` column missing — run traffic_labeling.py.")
    print(f"   Congestion rate:        {df['Traffic_Congestion'].mean() * 100:.1f}%")
    print(f"   High-risk accident rate: {df['High_Risk_Accident'].mean() * 100:.1f}%")

    # === Traffic ===
    print("\n--- Training Traffic model ---")
    missing = [c for c in TRAFFIC_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"TRAFFIC_FEATURES missing from dataset: {missing}")
    t_model, t_scaler = _train_one(
        df[TRAFFIC_FEATURES].copy(), df["Traffic_Congestion"], "Traffic"
    )
    joblib.dump(t_model, models_dir / "traffic_model.pkl")
    joblib.dump(t_scaler, models_dir / "scaler_traffic.pkl")
    joblib.dump(TRAFFIC_FEATURES, models_dir / "traffic_features.pkl")

    # === Severity ===
    print("\n--- Training Severity model ---")
    missing = [c for c in SEVERITY_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"SEVERITY_FEATURES missing from dataset: {missing}")
    s_model, s_scaler = _train_one(
        df[SEVERITY_FEATURES].copy(), df["High_Risk_Accident"], "Severity"
    )
    joblib.dump(s_model, models_dir / "severity_model.pkl")
    joblib.dump(s_scaler, models_dir / "scaler_severity.pkl")
    joblib.dump(SEVERITY_FEATURES, models_dir / "severity_features.pkl")

    print("\n" + "=" * 60)
    print(f" Saved both models to {models_dir}")
    print(f" Total time: {time.time() - start:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    train_models()
