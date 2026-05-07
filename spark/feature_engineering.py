"""
Feature engineering step of the TrafficIQ pipeline.

Reads:  data/processed/cleaned_data.csv
Writes: data/processed/featured_data.csv

Run:
    python spark/feature_engineering.py

Notes
-----
* Bug fixed: the original `extract_road_features` returned RANDOM
  Road_Width / Speed_Limit values for rows with a missing street name
  (`np.random.uniform(...)`). That made the pipeline non-deterministic and
  silently injected noise into both training and inference. We now return
  a stable default (Internal road, mid-range width and speed). Other branches
  also returned a single fixed value despite the comments mentioning ranges
  — that part was already deterministic and is preserved.
* `Is_Night` falls back to an Hour-based rule when `Sunrise_Sunset` is missing.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"


# Regex patterns — compiled once for speed.
HIGHWAY_RE = re.compile(
    r"\bI-\d+|\bUS-\d+|\bHWY\b|\bHIGHWAY\b|\bPKWY\b|\bEXPY\b|\bFWY\b|\bTPKE\b"
)
LOCAL_RE = re.compile(
    r"\bST\b|\bRD\b|\bDR\b|\bAVE\b|\bBLVD\b|\bLN\b|"
    r"\bSR-\d+|\bCR-\d+|\bFL-\d+|\bCA-\d+|\bTX-\d+"
)


def extract_road_features(street_name: object) -> tuple[str, float, float]:
    """Return (Road_Type, Road_Width(m), Speed_Limit(mph)) deterministically."""
    if pd.isna(street_name):
        # NOTE: previously this returned random uniforms — use a stable default.
        return "Internal", 18.5, 25.0

    s = str(street_name).upper()
    if HIGHWAY_RE.search(s):
        return "Highway", 34.5, 75.0
    if LOCAL_RE.search(s):
        return "Local", 22.5, 45.0
    return "Internal", 18.5, 25.0


def get_weather_type(w: object) -> int:
    """Coarse-grained weather encoding used by the rest of the pipeline."""
    s = str(w).lower()
    if any(k in s for k in ("rain", "drizzle", "shower", "storm", "thunder", "squall")):
        return 2  # Rain-like
    if any(k in s for k in ("snow", "sleet", "ice", "hail", "wintry", "grains")):
        return 3  # Snow-like
    if any(k in s for k in ("fog", "mist", "haze", "smoke", "dust", "sand", "ash", "whirlwind")):
        return 4  # Reduced visibility
    if any(k in s for k in ("cloud", "overcast")):
        return 4  # Treat overcast as reduced-visibility per project convention
    return 0  # Clear


ROAD_TYPE_MAP = {"Local": 1, "Internal": 2, "Highway": 3}


def run_feature_engineering(input_path: os.PathLike | str = INPUT_PATH,
                            output_path: os.PathLike | str = OUTPUT_PATH) -> Path:
    start_time = time.time()
    input_path = Path(input_path)
    output_path = Path(output_path)

    print("=" * 60)
    print("[FEATURE ENGINEERING MODULE] Starting...")
    print("=" * 60)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {input_path}. "
            "Run spark/data_cleaning.py first."
        )

    print("-> Loading cleaned data...")
    df = pd.read_csv(input_path, low_memory=False)

    # 1) Road type / width / speed
    print("-> Extracting Road Type, Width, Speed Limits...")
    if "Street" in df.columns:
        road_features = df["Street"].apply(
            lambda x: pd.Series(extract_road_features(x))
        )
    else:
        road_features = pd.DataFrame(
            [extract_road_features(np.nan)] * len(df),
            index=df.index,
        )
    road_features.columns = ["Road_Type", "Road_Width(m)", "Speed_Limit(mph)"]
    df[["Road_Type", "Road_Width(m)", "Speed_Limit(mph)"]] = road_features
    df = df.drop(columns=["Street"], errors="ignore")

    # 2) Time-based booleans
    if "Hour" not in df.columns:
        raise KeyError("Hour column missing — did you run data_cleaning.py first?")

    df["Is_RushHour"] = df["Hour"].apply(
        lambda h: 1 if (6 <= h <= 9) or (16 <= h <= 19) else 0
    )

    # Prefer Sunrise_Sunset; fall back to hour-based rule
    if "Sunrise_Sunset" in df.columns:
        df["Is_Night"] = df["Sunrise_Sunset"].apply(
            lambda x: 1 if str(x).strip().lower() == "night" else 0
        )
    else:
        df["Is_Night"] = df["Hour"].apply(lambda h: 1 if (h >= 21 or h <= 5) else 0)

    # 3) Weather encoding
    print("-> Simplifying weather categories...")
    if "Weather_Condition" in df.columns:
        df["Weather_Enc"] = df["Weather_Condition"].apply(get_weather_type)
    else:
        df["Weather_Enc"] = 0

    # 4) Road type encoding
    print("-> Encoding road type to numbers...")
    df["Road_Type_Enc"] = df["Road_Type"].map(ROAD_TYPE_MAP)

    # 5) Drop now-redundant text columns
    df = df.drop(
        columns=["Start_Time", "Sunrise_Sunset", "Road_Type", "Weather_Condition"],
        errors="ignore",
    )

    # 6) Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("-> Saving featured data...")
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"  Feature engineering finished in {elapsed:.2f}s")
    print(f"  Output saved to: {output_path}")
    print(f"  Final shape: {df.shape}\n")
    return output_path


if __name__ == "__main__":
    run_feature_engineering()
