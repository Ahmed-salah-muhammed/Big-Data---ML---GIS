"""
Data cleaning step of the TrafficIQ pipeline.

Reads:  data/raw/dirty_dataset.csv          (US Accidents 2016-2023)
Writes: data/processed/cleaned_data.csv

Run:
    python spark/data_cleaning.py

Notes
-----
* Despite the directory name `spark/`, this module is pure pandas — it has
  always been pandas in this project; the name is historical.
* Bug fixed in this file: the final summary print was a regular string with
  literal "{...}" placeholders (missing the `f` prefix), so the original
  output read literally "Data Cleaning Finished in {time.time()...}s".
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dirty_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "cleaned_data.csv"

# Columns that add no signal for accident / traffic prediction.
COLS_TO_DROP = [
    "HEAD", "ID", "Source", "End_Lat", "End_Lng", "End_Time",
    "Distance(mi)", "Description", "City", "County", "State", "Zipcode",
    "Country", "Timezone", "Weather_Timestamp", "Wind_Chill(F)",
    "Wind_Direction", "Civil_Twilight", "Nautical_Twilight",
    "Astronomical_Twilight", "Turning_Loop",
]

BOOL_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal",
]

WEATHER_COLS = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)",
]


def run_data_cleaning(raw_path: os.PathLike | str = RAW_PATH,
                      output_path: os.PathLike | str = OUTPUT_PATH) -> Path:
    """Clean the raw US-Accidents CSV and save a cleaned copy.

    Returns the path to the cleaned CSV.
    """
    start_time = time.time()
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[DATA CLEANING MODULE] Starting...")
    print("=" * 60)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}.\n"
            "Download the US Accidents (2016-2023) dataset from Kaggle and "
            "place it at data/raw/dirty_dataset.csv before running."
        )

    # 1) Load
    print("-> Loading raw data...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"   Loaded {len(df):,} records.")

    # 2) Drop useless columns
    existing = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing)
    print(f"   Dropped {len(existing)} useless columns.")

    # 3) Repair broken Start_Time values
    print("-> Fixing broken dates (e.g., '37:14.0')...")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Start_Time"])
    print(f"   Dropped {before - len(df):,} rows with unparseable Start_Time.")
    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month

    # 4) Booleans -> 0/1
    print("-> Converting boolean columns to 0/1...")
    bool_present = [c for c in BOOL_COLS if c in df.columns]
    df[bool_present] = df[bool_present].astype(int)

    # 5) Fill missing weather values: median per Airport_Code+Month, then global median
    print("-> Filling missing weather values (per-airport / per-month median)...")
    for col in WEATHER_COLS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if {"Airport_Code", "Month"}.issubset(df.columns):
            df[col] = (
                df.groupby(["Airport_Code", "Month"])[col]
                  .transform(lambda x: x.fillna(x.median()))
            )
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[c for c in WEATHER_COLS if c in df.columns])
    df = df.drop(columns=["Airport_Code"], errors="ignore")

    # 6) Precipitation imputation + missingness flag
    if "Precipitation(in)" in df.columns:
        df["Precipitation_NA"] = df["Precipitation(in)"].isna().astype(int)
        df["Precipitation(in)"] = df["Precipitation(in)"].fillna(
            df["Precipitation(in)"].median()
        )

    # 7) Save
    print("-> Saving cleaned data...")
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"  Data cleaning finished in {elapsed:.2f}s")
    print(f"  Output saved to: {output_path}")
    print(f"  Final shape: {df.shape}\n")
    return output_path


if __name__ == "__main__":
    run_data_cleaning()
