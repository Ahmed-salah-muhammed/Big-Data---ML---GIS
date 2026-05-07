"""
Traffic labeling step of the TrafficIQ pipeline.

Reads:  data/processed/featured_data.csv
Writes: data/processed/final_dataset.csv

Run:
    python spark/traffic_labeling.py

Notes
-----
* This module is the **single source of truth** for the `Traffic_Congestion`
  and `High_Risk_Accident` columns. The training script must NOT recompute
  them — the original train_model.py silently overrode the labels with a
  different formula, which was confusing.
* The original implementation called `df.apply(..., axis=1)` which is O(n)
  Python-level. The new version is vectorized — typically ~50x faster on the
  full 500k-row dataset.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"

CONGESTION_THRESHOLD = 4
SEVERITY_HIGH_RISK_THRESHOLD = 3


def run_traffic_labeling(input_path: os.PathLike | str = INPUT_PATH,
                         output_path: os.PathLike | str = OUTPUT_PATH) -> Path:
    start_time = time.time()
    input_path = Path(input_path)
    output_path = Path(output_path)

    print("=" * 60)
    print("[TRAFFIC LABELING MODULE] Starting...")
    print("=" * 60)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Featured dataset not found at {input_path}. "
            "Run spark/feature_engineering.py first."
        )

    print("-> Loading featured data...")
    df = pd.read_csv(input_path, low_memory=False)

    # ----- Vectorized congestion score ----------------------------------
    # Each rule contributes a fixed weight; sum them and threshold.
    print("-> Calculating Traffic_Congestion labels (vectorized)...")
    score = (
        ((df["Road_Width(m)"] < 24).astype(int) * 2)        # Narrow road
      + ((df["Is_RushHour"] == 1).astype(int) * 2)          # Rush hour
      + ((df["Speed_Limit(mph)"] < 45).astype(int) * 1)     # Urban speed limit
      + ((df["Is_Night"] == 0).astype(int) * 1)             # Daytime baseline
      + (df["Weather_Enc"].isin([2, 3, 4]).astype(int) * 2) # Bad weather
    )
    df["Traffic_Congestion"] = (score >= CONGESTION_THRESHOLD).astype(int)

    # ----- High-risk accident label -------------------------------------
    print("-> Calculating High_Risk_Accident labels...")
    if "Severity" not in df.columns:
        raise KeyError(
            "`Severity` column missing — it must come from the raw dataset "
            "and survive both data_cleaning.py and feature_engineering.py."
        )
    df["High_Risk_Accident"] = (df["Severity"] >= SEVERITY_HIGH_RISK_THRESHOLD).astype(int)

    # ----- Save ---------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("-> Saving final dataset...")
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    congested_pct = df["Traffic_Congestion"].mean() * 100
    high_risk_pct = df["High_Risk_Accident"].mean() * 100
    print(f"  Traffic labeling finished in {elapsed:.2f}s")
    print(f"  Output saved to: {output_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Stats: {congested_pct:.1f}% Congestion, "
          f"{high_risk_pct:.1f}% High Risk\n")
    return output_path


if __name__ == "__main__":
    run_traffic_labeling()
