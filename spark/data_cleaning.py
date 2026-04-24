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
        "dirty_dataset.csv",
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
        "End_Time",
        "Distance(mi)",
        "Description",
        "City",
        "County",
        "State",
        "Zipcode",
        "Country",
        "Timezone",
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

    # Drop rows where Start_Time couldn't be parsed
    df = df.dropna(subset=["Start_Time"])

    # Extract Hour and Month for later use 
    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month  

    # 4. Convert True/False to 1/0 
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
    df[bool_cols] = df[bool_cols].astype(int)

    # 5. Fill Missing Numeric Values with Median 
    print("-> Filling missing numeric values with median, grouped by location...")
    weather_cols = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df.groupby(['Airport_Code','Month'])[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    # Drop any remaining rows with missing weather data just in case
    df=df.dropna(subset=weather_cols)
    df=df.drop(columns=['Airport_Code'], errors='ignore')  # Drop Airport_Code if it exists, since we won't use it

    df['Precipitation_NA'] = 0
    df.loc[df['Precipitation(in)'].isnull(),'Precipitation_NA'] = 1
    df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].median())  


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
