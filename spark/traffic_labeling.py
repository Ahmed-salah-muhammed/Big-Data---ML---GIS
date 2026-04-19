import pandas as pd
import os
import time


def run_traffic_labeling():
    start_time = time.time()

    # Define paths
    input_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "featured_data.csv"
    )
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "final_dataset.csv"
    )

    print("=" * 60)
    print("[TRAFFIC LABELING MODULE] Starting...")
    print("=" * 60)

    # 1. Load Featured Data
    print("-> Loading featured data...")
    df = pd.read_csv(input_path, low_memory=False)

    # 2. Calculate Traffic Congestion Target
    # Based on your rule: Road Width + Rush Hour + Weather
    print("-> Calculating Traffic Congestion labels...")

    def calc_congestion(row):
        score = 0
        # If road is narrow (Local or Internal, less than 24m)
        if row["Road_Width(m)"] < 24:
            score += 2
        # If it's Rush Hour
        if row["Is_RushHour"] == 1:
            score += 2
        # If speed limit is low (means urban/local area) -> +1 point
        if row["Speed_Limit(mph)"] < 45:
            score += 1
        # Daytime has slightly more baseline traffic than night
        if row["Is_Night"] == 0:
            score += 1
        # Bad weather increases congestion probability
        if row["Weather_Enc"] in [2, 3, 4]:
            score += 2

        # If score is high, label as 1 (Congested)
        return 1 if score >= 4 else 0

    df["Traffic_Congestion"] = df.apply(calc_congestion, axis=1)

    # 3. Calculate Accident Severity Target
    # If severity is 3 or 4, it's High Risk (1), else Low Risk (0)
    print("-> Calculating High Risk Accident labels...")
    df["High_Risk_Accident"] = df["Severity"].apply(lambda x: 1 if x >= 3 else 0)

    # 4. Save Final Dataset
    print("-> Saving final dataset...")
    df.to_csv(output_path, index=False)

    # Quick stats
    congested_pct = df["Traffic_Congestion"].mean() * 100
    highrisk_pct = df["High_Risk_Accident"].mean() * 100

    print(f" Traffic Labeling Finished in {time.time() - start_time:.2f}s")
    print(" Output saved to: data/processed/final_dataset.csv")
    print(f" Final Shape: {df.shape}")
    print(f" Stats: {congested_pct:.1f}% Congestion, {highrisk_pct:.1f}% High Risk\n")


if __name__ == "__main__":
    run_traffic_labeling()
