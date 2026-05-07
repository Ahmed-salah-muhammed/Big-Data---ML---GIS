"""
Train the spatiotemporal traffic-density model.

Reads:  data/processed/spatiotemporal_dataset.csv  (from spatiotemporal_labeling.py)
Writes: models/traffic_st_model.pkl
        models/traffic_st_features.pkl

Run:
    python models/train_traffic_spatiotemporal.py

What's different from `train_model.py`
--------------------------------------
* The label `Traffic_Level` ∈ {0=Low, 1=Medium, 2=High} is the **global
  tertile of the empirical accident-count distribution per (cell, hour)
  bucket** — not a deterministic rule over the row's features. So the model
  has something real to learn.
* Features are **leakage-free**: spatial (cell-mean lat/lng), temporal
  (hour, mean-month), aggregated environmental conditions, infrastructure
  fractions, and *cell baselines* (how active the cell is overall). Crucially
  `accident_count` is excluded — that IS the label.
* Two evaluation regimes are reported:
    1. **Random 5-fold (StratifiedKFold)** — does the model generalize to
       new (cell, hour) buckets in cells we've seen before? Tests temporal
       and conditional generalization.
    2. **Spatial 5-fold (GroupKFold by Cell_ID)** — does it generalize to
       *unseen cells*? Tests true spatial generalization. This number is
       expected to be lower than (1); the gap quantifies how much the model
       relies on memorizing per-cell baselines vs. learning transferable
       patterns.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "spatiotemporal_dataset.csv"
MODELS_DIR = HERE

# ---------------------------------------------------------------------------
# Features  — leakage-free
# ---------------------------------------------------------------------------
# These are the columns we feed to the model. Notable exclusions:
#   - accident_count        (the label is derived from it — would be perfect leakage)
#   - Cell_ID               (would let the model memorize per-cell labels and
#                            also breaks under spatial holdout — no shared IDs)
#   - Traffic_Level         (the label itself)
SPATIAL_FEATURES   = ["Cell_Lat_Mean", "Cell_Lng_Mean"]
TEMPORAL_FEATURES  = ["Hour", "mean_Month"]
WEATHER_FEATURES   = [
    "mean_Temperature_F_", "mean_Humidity_Pct_", "mean_Pressure_in_",
    "mean_Visibility_mi_", "mean_Wind_Speed_mph_", "mean_Precipitation_in_",
    "mean_Weather_Enc",
]
ROAD_FEATURES      = ["mean_Road_Type_Enc"]
INFRA_FEATURES     = [
    "frac_Junction", "frac_Traffic_Signal", "frac_Crossing",
    "frac_Stop", "frac_Roundabout", "frac_Station", "frac_Amenity",
]
BASELINE_FEATURES  = ["cell_total_accidents", "cell_unique_hours", "cell_unique_months"]

ALL_FEATURES = (
    SPATIAL_FEATURES + TEMPORAL_FEATURES + WEATHER_FEATURES
    + ROAD_FEATURES + INFRA_FEATURES + BASELINE_FEATURES
)

XGB_KWARGS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
)


def _summarize_fold(y_true, y_pred, label: str) -> dict:
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    rec_m = recall_score(y_true, y_pred, average="macro")
    acc = (y_true == y_pred).mean()
    print(f"   [{label}] acc={acc:.3f}  F1-macro={f1m:.3f}  "
          f"F1-weighted={f1w:.3f}  recall-macro={rec_m:.3f}")
    return {"acc": acc, "f1_macro": f1m, "f1_weighted": f1w, "recall_macro": rec_m}


def evaluate_random_kfold(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[dict]:
    print("\n" + "=" * 60)
    print(f"[1/2] Random Stratified {n_splits}-Fold")
    print("    (Tests: can we predict new buckets in cells we've trained on?)")
    print("=" * 60)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        m = xgb.XGBClassifier(**XGB_KWARGS)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        preds = m.predict(X.iloc[te])
        folds.append(_summarize_fold(y.iloc[te].values, preds, f"random fold {i}"))
    return folds


def evaluate_spatial_groupkfold(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                                n_splits: int = 5) -> list[dict]:
    print("\n" + "=" * 60)
    print(f"[2/2] Spatial Group {n_splits}-Fold (held-out CELLS)")
    print("    (Tests: can we predict in cells the model has NEVER seen?)")
    print("=" * 60)
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        train_cells = groups.iloc[tr].nunique()
        test_cells  = groups.iloc[te].nunique()
        m = xgb.XGBClassifier(**XGB_KWARGS)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        preds = m.predict(X.iloc[te])
        folds.append(_summarize_fold(y.iloc[te].values, preds,
                                     f"spatial fold {i} (train_cells={train_cells:,}, test_cells={test_cells:,})"))
    return folds


def _mean_std(folds: list[dict], key: str) -> tuple[float, float]:
    vals = np.array([f[key] for f in folds])
    return float(vals.mean()), float(vals.std())


def train_full_and_save(X: pd.DataFrame, y: pd.Series,
                        models_dir: Path = MODELS_DIR) -> None:
    print("\n" + "=" * 60)
    print("Training final model on the full dataset...")
    print("=" * 60)
    model = xgb.XGBClassifier(**XGB_KWARGS)
    model.fit(X, y, verbose=False)

    # Diagnostic on training set (NOT a generalization measure)
    preds = model.predict(X)
    print("\n--- Final model — training-set diagnostic ---")
    print(classification_report(y, preds, target_names=["Low", "Medium", "High"], digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))

    # Top features for explainability
    fi = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print("\nTop 10 features by importance:")
    for name, val in fi.head(10).items():
        bar = "█" * int(val * 200)
        print(f"  {name:>30} {val:6.3f}  {bar}")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "traffic_st_model.pkl")
    joblib.dump(ALL_FEATURES, models_dir / "traffic_st_features.pkl")
    print(f"\n  Saved: {models_dir / 'traffic_st_model.pkl'}")
    print(f"  Saved: {models_dir / 'traffic_st_features.pkl'}")


def train_spatiotemporal(input_path: os.PathLike | str = INPUT_PATH,
                         models_dir: os.PathLike | str = MODELS_DIR) -> None:
    start = time.time()
    input_path = Path(input_path)
    models_dir = Path(models_dir)

    print("=" * 60)
    print("[SPATIOTEMPORAL TRAFFIC] Training start")
    print("=" * 60)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run spark/spatiotemporal_labeling.py first."
        )

    df = pd.read_csv(input_path, low_memory=False)
    print(f"-> Loaded {len(df):,} buckets across {df['Cell_ID'].nunique():,} cells.")

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Features missing from dataset: {missing}")

    X = df[ALL_FEATURES].copy()
    y = df["Traffic_Level"].astype(int)
    groups = df["Cell_ID"]

    print(f"   Features ({len(ALL_FEATURES)}): {', '.join(ALL_FEATURES[:6])}, ...")
    print(f"   Class balance: {y.value_counts().sort_index().to_dict()}")

    # ------------------ Evaluation -----------------------------------------
    rand_folds = evaluate_random_kfold(X, y)
    spat_folds = evaluate_spatial_groupkfold(X, y, groups)

    # ------------------ Summary --------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std across 5 folds)")
    print("=" * 60)
    for label, folds in [("Random       ", rand_folds), ("Spatial-group", spat_folds)]:
        mF1, sF1 = _mean_std(folds, "f1_macro")
        mAcc, sAcc = _mean_std(folds, "acc")
        mRec, sRec = _mean_std(folds, "recall_macro")
        print(f"  {label}  acc={mAcc:.3f}±{sAcc:.3f}  "
              f"F1-macro={mF1:.3f}±{sF1:.3f}  recall-macro={mRec:.3f}±{sRec:.3f}")

    rand_f1, _ = _mean_std(rand_folds, "f1_macro")
    spat_f1, _ = _mean_std(spat_folds, "f1_macro")
    gap = rand_f1 - spat_f1
    print(f"\n  Spatial generalization gap (random F1 − spatial F1) = {gap:.3f}")
    if gap > 0.10:
        print("  → Large gap: the model relies heavily on per-cell baseline "
              "signal that doesn't transfer to new cells. Consider adding "
              "transferable spatial features (e.g., distance to city center, "
              "POI density).")
    else:
        print("  → Small gap: the model has learned transferable patterns.")

    # ------------------ Final fit & save -----------------------------------
    train_full_and_save(X, y, models_dir=models_dir)

    print(f"\n  Total time: {time.time() - start:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    train_spatiotemporal()
