"""
Train geo-agnostic versions of the severity and spatiotemporal-traffic models.

Why this script exists
----------------------
The default models (severity_model.pkl, traffic_st_model.pkl) include geographic
features — Start_Lat / Start_Lng for severity, plus Cell_Lat_Mean / Cell_Lng_Mean
and `cell_total_accidents` / `cell_unique_hours` / `cell_unique_months` for the
spatiotemporal model. Those features are extremely useful inside the US, where
the training data lives, but they don't transfer to other countries:

* Cairo's longitude (+31°) is outside the US training range (-125° to -66°).
  XGBoost extrapolates by falling off the last decision-tree leaf — silent
  garbage rather than an error.
* The cell-baseline features are computed from a US cell lookup table; an
  Egyptian (lat, lng) maps to a cell with no entry, falls back to the global
  US median, and ends up identical for every Cairo point.

This script retrains both models on the SAME data with the SAME label, but
drops the geographic columns from the feature set. The result:

    severity_model_geo_agnostic.pkl       — 26 features, no lat/lng
    scaler_severity_geo_agnostic.pkl
    severity_features_geo_agnostic.pkl

    traffic_st_model_geo_agnostic.pkl     — 15 features, no spatial info
    traffic_st_features_geo_agnostic.pkl

The API loads these and uses them automatically when the request location is
outside the continental US. Inside the US it keeps using the geo-aware models
(better F1).

Run:
    python models/train_geo_agnostic.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

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
SEVERITY_INPUT = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
ST_INPUT = PROJECT_ROOT / "data" / "processed" / "spatiotemporal_dataset.csv"
MODELS_DIR = HERE


# ---------------------------------------------------------------------------
# Feature lists  — geographic columns intentionally removed
# ---------------------------------------------------------------------------
# Severity: same as train_model.SEVERITY_FEATURES MINUS Start_Lat / Start_Lng
SEVERITY_FEATURES_GEO_AGNOSTIC = [
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

# Spatiotemporal: same as train_traffic_spatiotemporal.ALL_FEATURES MINUS
# Cell_Lat_Mean, Cell_Lng_Mean, cell_total_accidents, cell_unique_hours,
# cell_unique_months. What's left is purely conditions + temporal + infra.
ST_FEATURES_GEO_AGNOSTIC = [
    "Hour",
    "mean_Month",
    "mean_Temperature_F_",
    "mean_Humidity_Pct_",
    "mean_Pressure_in_",
    "mean_Visibility_mi_",
    "mean_Wind_Speed_mph_",
    "mean_Precipitation_in_",
    "mean_Weather_Enc",
    "mean_Road_Type_Enc",
    "frac_Junction",
    "frac_Traffic_Signal",
    "frac_Crossing",
    "frac_Stop",
    "frac_Roundabout",
    "frac_Station",
    "frac_Amenity",
]

XGB_BINARY = dict(
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
XGB_MULTI = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    random_state=42,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Severity (binary)
# ---------------------------------------------------------------------------
def train_severity_geo_agnostic() -> None:
    print("=" * 60)
    print("[GEO-AGNOSTIC SEVERITY] training...")
    print("=" * 60)
    if not SEVERITY_INPUT.exists():
        raise FileNotFoundError(f"{SEVERITY_INPUT} not found.")

    df = pd.read_csv(SEVERITY_INPUT, low_memory=False).dropna()
    missing = [c for c in SEVERITY_FEATURES_GEO_AGNOSTIC if c not in df.columns]
    if missing:
        raise KeyError(f"Severity features missing: {missing}")

    X = df[SEVERITY_FEATURES_GEO_AGNOSTIC].copy()
    y = df["High_Risk_Accident"].astype(int)
    print(
        f"-> {len(df):,} records, {len(SEVERITY_FEATURES_GEO_AGNOSTIC)} features, "
        f"positive rate {y.mean()*100:.1f}%"
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler: StandardScaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    pos = max(int((y_tr == 1).sum()), 1)
    neg = int((y_tr == 0).sum())
    pos_w = neg / pos
    model = xgb.XGBClassifier(scale_pos_weight=pos_w, **XGB_BINARY)
    model.fit(X_tr_s, y_tr)

    preds = model.predict(X_te_s)
    print(classification_report(y_te, preds, digits=3))

    joblib.dump(model, MODELS_DIR / "severity_model_geo_agnostic.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler_severity_geo_agnostic.pkl")
    joblib.dump(
        SEVERITY_FEATURES_GEO_AGNOSTIC,
        MODELS_DIR / "severity_features_geo_agnostic.pkl",
    )
    print(f"-> Saved severity_model_geo_agnostic.pkl")


# ---------------------------------------------------------------------------
# Spatiotemporal (3-class)
# ---------------------------------------------------------------------------
def train_spatiotemporal_geo_agnostic() -> None:
    print("\n" + "=" * 60)
    print("[GEO-AGNOSTIC SPATIOTEMPORAL] training + 5-fold CV...")
    print("=" * 60)
    if not ST_INPUT.exists():
        raise FileNotFoundError(
            f"{ST_INPUT} not found. Run spark/spatiotemporal_labeling.py first."
        )

    df = pd.read_csv(ST_INPUT, low_memory=False)
    missing = [c for c in ST_FEATURES_GEO_AGNOSTIC if c not in df.columns]
    if missing:
        raise KeyError(f"ST features missing: {missing}")

    X = df[ST_FEATURES_GEO_AGNOSTIC].copy()
    y = df["Traffic_Level"].astype(int)
    print(
        f"-> {len(df):,} buckets, {len(ST_FEATURES_GEO_AGNOSTIC)} features, "
        f"class balance {y.value_counts().sort_index().to_dict()}"
    )

    # 5-fold stratified CV — to compare honestly with the geo-aware model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        m = xgb.XGBClassifier(**XGB_MULTI)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        preds = m.predict(X.iloc[te])
        f1m = f1_score(y.iloc[te], preds, average="macro")
        f1s.append(f1m)
        print(f"   fold {i}  F1-macro={f1m:.3f}")
    print(f"   mean F1-macro = {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"   (geo-aware model scored 0.811 ± 0.003 on the same data)")

    # Train final model on full data
    model = xgb.XGBClassifier(**XGB_MULTI)
    model.fit(X, y, verbose=False)

    # Feature importance
    fi = pd.Series(
        model.feature_importances_, index=ST_FEATURES_GEO_AGNOSTIC
    ).sort_values(ascending=False)
    print("\nTop 8 features by importance (geo-agnostic model):")
    for n, v in fi.head(8).items():
        bar = "█" * int(v * 200)
        print(f"  {n:>28} {v:6.3f}  {bar}")

    joblib.dump(model, MODELS_DIR / "traffic_st_model_geo_agnostic.pkl")
    joblib.dump(
        ST_FEATURES_GEO_AGNOSTIC, MODELS_DIR / "traffic_st_features_geo_agnostic.pkl"
    )
    print(f"\n-> Saved traffic_st_model_geo_agnostic.pkl")


if __name__ == "__main__":
    start = time.time()
    train_severity_geo_agnostic()
    train_spatiotemporal_geo_agnostic()
    print(f"\n{'='*60}\nTotal time: {time.time()-start:.1f}s\n{'='*60}")
