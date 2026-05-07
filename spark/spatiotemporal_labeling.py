"""
Spatiotemporal labeling — empirical density replacement for the rule-based
`Traffic_Congestion` column.

Reads:  data/processed/final_dataset.csv  (columns from the existing pipeline)
Writes: data/processed/spatiotemporal_dataset.parquet  (or .csv as fallback)

Run:
    python spark/spatiotemporal_labeling.py

Why this module exists
----------------------
The original `traffic_labeling.py` defines the label as a deterministic rule
over `Road_Width(m)`, `Is_RushHour`, `Speed_Limit(mph)`, `Is_Night`, and
`Weather_Enc` — and then `train_model.py` hands all five of those columns to
the model as features. The model recovers the rule and reports ~100 % accuracy.

This module replaces the rule with an *empirical* label drawn from the data
itself: for each (spatial cell, hour-of-day, month) bucket, we count the
number of recorded accidents and bucket the global count distribution into
three classes — Low / Medium / High traffic. The label is now a property of
the aggregate observation, not of the individual row's predictors, so the
model has to learn real spatiotemporal structure.

Pipeline
--------
1. Spatial discretization — quantize lat/lng to a fixed cell size.
2. Temporal aggregation — group by (Cell_ID, Hour, Month).
3. Aggregate features — means of per-record conditions, max of infrastructure
   flags, plus *cell baseline* signals (total accidents in the cell across all
   time, number of distinct active hours).
4. Global tertile labels — Low / Medium / High = bottom / middle / top third
   of the global `accident_count` distribution.
5. Drop sparsely-observed cells (configurable minimum).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "spatiotemporal_dataset.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CELL_SIZE_DEG = 0.05          # ~5.5 km lat × ~4.4 km lng — neighborhood scale.
                              # 0.01° was tested and produced a degenerate
                              # count distribution (92% of buckets had count=1)
                              # which makes the tertile labels meaningless.
MIN_OBS_PER_CELL = 5          # drop cells with fewer than this many buckets
N_TRAFFIC_CLASSES = 3         # Low / Medium / High
GROUP_KEYS = ["Cell_ID", "Hour"]   # aggregation grain. Month would over-split
                              # this dataset; we keep it as an aggregated
                              # feature instead (see `mean_Month`).

# Per-record features we'll aggregate to the (cell, hour, month) level.
# These are leakage-free: they're conditions/infra at the time of the row,
# not properties of the aggregate count.
WEATHER_FEATURES = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Weather_Enc",
]
INFRA_FEATURES = [
    "Junction", "Traffic_Signal", "Crossing", "Stop",
    "Roundabout", "Station", "Amenity",
]
ROAD_FEATURES = ["Road_Type_Enc"]


def add_cell_id(df: pd.DataFrame, cell_size: float = CELL_SIZE_DEG) -> pd.DataFrame:
    """Quantize lat/lng to a discrete cell ID.

    The cell ID is `<lat_idx>_<lng_idx>` so it's stable, human-readable, and
    can be split back out for spatial cross-validation.
    """
    df = df.copy()
    df["Cell_Lat_Idx"] = np.round(df["Start_Lat"] / cell_size).astype(int)
    df["Cell_Lng_Idx"] = np.round(df["Start_Lng"] / cell_size).astype(int)
    df["Cell_ID"] = (
        df["Cell_Lat_Idx"].astype(str) + "_" + df["Cell_Lng_Idx"].astype(str)
    )
    return df


def aggregate_to_buckets(df: pd.DataFrame,
                         group_keys: list[str] = None) -> pd.DataFrame:
    """Group records into spatiotemporal buckets and aggregate features.

    `group_keys` defaults to (Cell_ID, Hour). Month is aggregated as a feature
    (`mean_Month`) rather than used as a grouping dimension because including
    it splits the dataset too finely — most (cell, hour, month) buckets end up
    with count=1, which collapses the quantile labels.
    """
    if group_keys is None:
        group_keys = GROUP_KEYS

    available_weather = [c for c in WEATHER_FEATURES if c in df.columns]
    available_infra   = [c for c in INFRA_FEATURES   if c in df.columns]
    available_road    = [c for c in ROAD_FEATURES    if c in df.columns]

    grp = df.groupby(group_keys, sort=False)

    agg_specs: dict[str, tuple[str, str]] = {
        "accident_count": ("Start_Lat", "size"),
        "Cell_Lat_Mean":  ("Start_Lat", "mean"),
        "Cell_Lng_Mean":  ("Start_Lng", "mean"),
    }
    if "Month" in df.columns and "Month" not in group_keys:
        # Use month as an aggregated feature when it's not a grouping key.
        agg_specs["mean_Month"] = ("Month", "mean")
    for c in available_weather + available_road:
        safe = c.replace("(", "_").replace(")", "_").replace("%", "Pct").replace("/", "_")
        agg_specs[f"mean_{safe}"] = (c, "mean")
    for c in available_infra:
        agg_specs[f"frac_{c}"] = (c, "mean")

    agg = grp.agg(**agg_specs).reset_index()

    # Cell-level baselines — computed once per cell, broadcast back
    cell_stats = (
        df.groupby("Cell_ID")
          .agg(cell_total_accidents=("Start_Lat", "size"),
               cell_unique_hours=("Hour", "nunique"),
               cell_unique_months=("Month", "nunique"))
          .reset_index()
    )
    agg = agg.merge(cell_stats, on="Cell_ID", how="left")

    return agg


def assign_global_tertiles(agg: pd.DataFrame,
                           n_classes: int = N_TRAFFIC_CLASSES) -> pd.DataFrame:
    """Bucket `accident_count` into `n_classes` global quantile classes.

    We rank-then-qcut so ties at low counts (e.g., many buckets with count=1)
    get spread evenly rather than collapsing the lowest bin.
    """
    agg = agg.copy()
    ranks = agg["accident_count"].rank(method="first")
    agg["Traffic_Level"] = pd.qcut(
        ranks, q=n_classes, labels=list(range(n_classes))
    ).astype(int)
    return agg


def filter_sparse_cells(agg: pd.DataFrame,
                        min_obs: int = MIN_OBS_PER_CELL) -> pd.DataFrame:
    """Drop cells with too few (hour, month) observations to be reliable."""
    sizes = agg.groupby("Cell_ID").size()
    keep = sizes[sizes >= min_obs].index
    before = len(agg)
    agg = agg[agg["Cell_ID"].isin(keep)].copy()
    print(f"   Dropped {before - len(agg):,} rows from "
          f"{(sizes < min_obs).sum():,} sparse cells "
          f"(< {min_obs} buckets each); kept {len(agg):,} rows "
          f"across {agg['Cell_ID'].nunique():,} cells.")
    return agg


def build_spatiotemporal_dataset(
    input_path: os.PathLike | str = INPUT_PATH,
    output_path: os.PathLike | str = OUTPUT_PATH,
    cell_size: float = CELL_SIZE_DEG,
    min_obs_per_cell: int = MIN_OBS_PER_CELL,
    n_classes: int = N_TRAFFIC_CLASSES,
) -> Path:
    start = time.time()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[SPATIOTEMPORAL LABELING] Starting...")
    print("=" * 60)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run the standard pipeline first "
            "(data_cleaning.py → feature_engineering.py → traffic_labeling.py)."
        )

    print(f"-> Loading {input_path.name}...")
    df = pd.read_csv(input_path, low_memory=False).dropna(
        subset=["Start_Lat", "Start_Lng", "Hour", "Month"]
    )
    print(f"   Loaded {len(df):,} records.")

    print(f"-> Quantizing to {cell_size}° cells...")
    df = add_cell_id(df, cell_size=cell_size)
    print(f"   {df['Cell_ID'].nunique():,} unique cells.")

    print("-> Aggregating to (Cell, Hour, Month) buckets...")
    agg = aggregate_to_buckets(df)
    print(f"   {len(agg):,} buckets total.")

    print(f"-> Filtering cells with < {min_obs_per_cell} buckets...")
    agg = filter_sparse_cells(agg, min_obs=min_obs_per_cell)

    print(f"-> Assigning {n_classes}-class global tertile labels...")
    agg = assign_global_tertiles(agg, n_classes=n_classes)
    counts = agg["Traffic_Level"].value_counts().sort_index()
    pct = (counts / len(agg) * 100).round(1)
    print(f"   Class distribution (raw counts → % of dataset):")
    for cls, n in counts.items():
        print(f"     class {cls}  ({['Low','Medium','High'][cls]:>6}):  "
              f"{n:>8,}  ({pct[cls]}%)")

    # Quick sanity check: the count-per-class accident_count ranges
    print(f"   accident_count by class:")
    print(agg.groupby("Traffic_Level")["accident_count"]
             .describe()[["min", "50%", "max"]].to_string())

    print(f"-> Saving to {output_path}...")
    agg.to_csv(output_path, index=False)

    # Also save a compact cell baseline lookup for use at inference time.
    # The API uses this to translate a clicked (lat, lng) into the cell
    # baseline features the model expects.
    lookup_path = output_path.with_name("cell_baseline_lookup.csv")
    cell_lookup = (
        agg[["Cell_ID", "Cell_Lat_Mean", "Cell_Lng_Mean",
             "cell_total_accidents", "cell_unique_hours", "cell_unique_months"]]
        .drop_duplicates(subset=["Cell_ID"])
        .reset_index(drop=True)
    )
    cell_lookup.to_csv(lookup_path, index=False)
    print(f"  Cell baseline lookup → {lookup_path}  ({len(cell_lookup):,} cells)")

    # Persist the cell size used so the API can rebuild Cell_IDs from raw
    # lat/lng without guessing.
    config_path = output_path.with_name("spatiotemporal_config.json")
    import json
    with open(config_path, "w") as fh:
        json.dump({
            "cell_size_deg": cell_size,
            "min_obs_per_cell": min_obs_per_cell,
            "n_traffic_classes": n_classes,
            "group_keys": GROUP_KEYS,
        }, fh, indent=2)
    print(f"  Pipeline config → {config_path}")

    print(f"\n  Done in {time.time() - start:.2f}s")
    print(f"  Output shape: {agg.shape}")
    print(f"  Columns: {list(agg.columns)}")
    return output_path


if __name__ == "__main__":
    build_spatiotemporal_dataset()
