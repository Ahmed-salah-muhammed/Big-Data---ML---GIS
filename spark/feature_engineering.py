import pandas as pd
import numpy as np
import re
import os
import time


def run_feature_engineering():
    start_time = time.time()

    # Define paths
    input_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "cleaned_data.csv"
    )
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "featured_data.csv"
    )

    print("=" * 60)
    print("[FEATURE ENGINEERING MODULE] Starting...")
    print("=" * 60)

    # 1. Load Cleaned Data
    print("-> Loading cleaned data...")
    df = pd.read_csv(input_path, low_memory=False)

    # 2. Extract Road Type, Road Width Ranges, and Speed Limit
    print("-> Extracting Road Type, Width Ranges, and Speed Limits...")

    def extract_road_features(street_name):
        if pd.isna(street_name):
            return "Internal", np.random.uniform(16, 21), np.random.uniform(20, 30)

        street_upper = str(street_name).upper()

        # Highway: 32m to 36m width, 65 to 75 mph speed
        if re.search(
            r"\bI-\d+|\bUS-\d+|\bHWY\b|\bHIGHWAY\b|\bPKWY\b|\bEXPY\b|\bFWY\b|\bTPKE\b",
            street_upper,
        ):
            return "Highway", np.random.uniform(32, 36), np.random.uniform(80, 120)

        # Local: 21m to 24m width, 35 to 45 mph speed
        elif re.search(
            r"\bST\b|\bRD\b|\bDR\b|\bAVE\b|\bBLVD\b|\bLN\b|\bSR-\d+|\bCR-\d+|\bFL-\d+|\bCA-\d+|\bTX-\d+",
            street_upper,
        ):
            return "Local", np.random.uniform(21, 24), np.random.uniform(40, 70)

        # Internal: 16m to 21m width, 20 to 30 mph speed
        else:
            return "Internal", np.random.uniform(16, 21), np.random.uniform(20, 30)

    # Apply function to create the 3 new columns
    df[["Road_Type", "Road_Width(m)", "Speed_Limit(mph)"]] = df["Street"].apply(
        lambda x: pd.Series(extract_road_features(x))
    )
    df = df.drop(columns=["Street"])

    # Rush Hour: Morning (6-9) and Evening (16-19)
    df["Is_RushHour"] = df["Hour"].apply(
        lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
    )

    # Night detection
    df["Is_Night"] = df["Sunrise_Sunset"].apply(
        lambda x: 1 if str(x).lower() == "night" else 0
    )

    # 3. Simplify Weather Categories
    print("-> Simplifying Weather categories...")

    def get_weather_type(w):
        w = str(w).lower()
        if any(x in w for x in ['rain', 'drizzle', 'shower', 'storm', 'thunder', 'squall']): return 2
        if any(x in w for x in ['snow', 'sleet', 'ice', 'hail', 'wintry', 'grains']): return 3
        if any(x in w for x in ['fog', 'mist', 'haze', 'smoke', 'dust', 'sand', 'ash', 'whirlwind']): return 4
        if any(x in w for x in ['cloud', 'overcast']): return 4
        return 0 # Clear

    df['Weather_Enc'] = df['Weather_Condition'].apply(get_weather_type)


    # 4. Map Text Categories to Numbers
    print("-> Encoding text to numbers...")
    road_type_map = {
        "Local": 1,
        "Internal": 2,
        "Highway": 3,
    }

    df["Road_Type_Enc"] = df["Road_Type"].map(road_type_map)

    # Drop raw text columns that we just encoded
    df = df.drop(
        columns=[
            "Start_Time",
            "Sunrise_Sunset",
            "Road_Type",
            "Weather_Condition",
        ],
        errors="ignore",
    )

    # 5. Save Featured Data
    print("-> Saving featured data...")
    df.to_csv(output_path, index=False)

    print(f" Feature Engineering Finished in {time.time() - start_time:.2f}s")
    print(" Output saved to: data/processed/featured_data.csv")
    print(f" Final Shape: {df.shape}\n")



if __name__ == "__main__":
    run_feature_engineering()
