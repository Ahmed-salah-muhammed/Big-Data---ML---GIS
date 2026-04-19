import pandas as pd
import os
import time


def run_data_cleaning():
    # Start timer
    start_time = time.time()

    # Define paths based on the project structure
    raw_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "raw",
        "US_Accidents_March23_sampled_500k.csv",
    )
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    output_path = os.path.join(processed_dir, "cleaned_data.csv")

    os.makedirs(processed_dir, exist_ok=True)

    print("=" * 60)
    print("[DATA CLEANING MODULE] Starting...")
    print("=" * 60)

    # 1. Load Data
    print("-> Loading raw data...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"   Loaded {len(df):,} records.")

    # 2. Drop Useless Columns
    cols_to_drop = [
        "HEAD",
        "ID",
        "Source",
        "End_Lat",
        "End_Lng",
        "Distance(mi)",
        "Description",
        "City",
        "County",
        "State",
        "Zipcode",
        "Country",
        "Timezone",
        "Airport_Code",
        "Weather_Timestamp",
        "Wind_Chill(F)",
        "Wind_Direction",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
        "Turning_Loop",
    ]
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols)
    print(f"   Dropped {len(existing_cols)} useless columns.")

    # 3. Fix Broken Dates (Instead of dropping)
    print("-> Fixing broken dates (e.g., 37:14.0)...")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

    missing_dates = df["Start_Time"].isna().sum()
    if missing_dates > 0:
        # Calculate median hour and month from the VALID dates to create a logical dummy time
        median_hour = int(df["Start_Time"].dt.hour.median())
        median_month = int(df["Start_Time"].dt.month.median())
        dummy_date_str = f"2023-{median_month:02d}-01 {median_hour:02d}:00:00"
        dummy_date = pd.to_datetime(dummy_date_str)

        # Fill broken dates with the dummy date
        df["Start_Time"] = df["Start_Time"].fillna(dummy_date)  # 500000
        print(
            f"   Fixed {missing_dates} broken dates using median time: {dummy_date_str}"
        )

    # 4. Convert True/False to 1/0 (Avoiding Pandas ChainedAssignmentError)
    print("-> Converting boolean columns to 1/0...")
    bool_cols = [
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
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() == "true" else 0)

    # 5. Fill Missing Numeric Values with Median (Avoiding Pandas ChainedAssignmentError)
    print("-> Filling missing numeric values with median...")
    num_cols = [
        "Temperature(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 6. Save Cleaned Data
    print("-> Saving cleaned data...")
    df.to_csv(output_path, index=False)

    print(" Data Cleaning Finished in {time.time() - start_time:.2f}s")
    print(" Output saved to: data/processed/cleaned_data.csv")
    print(f" Total Time: {time.time() - start_time:.2f} seconds")
    print(f" Final Shape: {df.shape}\n")

    return output_path


# If run directly, execute the function
if __name__ == "__main__":
    run_data_cleaning()
