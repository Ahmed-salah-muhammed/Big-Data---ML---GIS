import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_models():
    start_time = time.time()

    # Define paths based on your structure
    input_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "final_dataset.csv"
    )
    models_dir = os.path.join(os.path.dirname(__file__))

    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("[MODEL TRAINING MODULE] Starting...")
    print("=" * 60)

    # 1. Load Final Dataset
    print("-> Loading final dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   Loaded {len(df):,} records.")

    # Drop rows with any missing values just in case (for clean ML input)
    df = df.dropna()
    print(f"   Clean records for ML: {len(df):,}")

    # ==========================================
    # MODEL 1: Accident Risk Prediction
    # ==========================================
    print("\n--- Training Model 1: Accident Risk ---")

    accident_features = [
        "Start_Lat",
        "Start_Lng",
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

    X_acc = df[accident_features]
    y_acc = df["High_Risk_Accident"]

    # Split data (80% train, 20% test)
    X_acc_train, X_acc_test, y_acc_train, y_acc_test = train_test_split(
        X_acc, y_acc, test_size=0.2, random_state=42
    )

    # Scale features (Important for distance-based models, good practice for RF too)
    scaler_acc = StandardScaler()
    X_acc_train_scaled = scaler_acc.fit_transform(X_acc_train)
    X_acc_test_scaled = scaler_acc.transform(X_acc_test)

    # Train Random Forest (n_jobs=-1 uses all CPU cores to speed up 500k rows)
    model_acc = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # Helps if most accidents are severity 2 (Low Risk)
    )
    model_acc.fit(X_acc_train_scaled, y_acc_train)

    preds_acc = model_acc.predict(X_acc_test_scaled)

    # 2. Print the report table you want
    print("\n--- Accident Risk Classification Report ---")
    print(classification_report(y_acc_test, preds_acc))


    # Evaluate
    preds_acc = model_acc.predict(X_acc_test_scaled)
    acc_score = accuracy_score(y_acc_test, preds_acc)
    print(f" Accident Model Accuracy: {acc_score * 100:.2f}%")

    # Save Model 1 artifacts
    joblib.dump(model_acc, os.path.join(models_dir, "severity_model.pkl"))
    joblib.dump(scaler_acc, os.path.join(models_dir, "scaler_severity.pkl"))
    joblib.dump(accident_features, os.path.join(models_dir, "severity_features.pkl"))
    print("   Saved: severity_model.pkl, scaler_severity.pkl, severity_features.pkl")

    # ==========================================
    # MODEL 2: Traffic Congestion Prediction
    # ==========================================
    print("\n--- Training Model 2: Traffic Congestion ---")

    # Focusing on Location, Time, and Road properties
    traffic_features = [
        "Start_Lat",
        "Start_Lng",
        "Hour",
        "Month",
        "Is_RushHour",
        "Is_Night",
        "Road_Type_Enc",
        "Road_Width(m)",
        "Speed_Limit(mph)",
        "Weather_Enc",
        "Junction",
        "Traffic_Signal",
    ]

    X_traf = df[traffic_features]
    y_traf = df["Traffic_Congestion"]

    # Split data
    X_traf_train, X_traf_test, y_traf_train, y_traf_test = train_test_split(
        X_traf, y_traf, test_size=0.2, random_state=42
    )

    # Scale features
    scaler_traf = StandardScaler()
    X_traf_train_scaled = scaler_traf.fit_transform(X_traf_train)
    X_traf_test_scaled = scaler_traf.transform(X_traf_test)

    # Train Random Forest
    model_traf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model_traf.fit(X_traf_train_scaled, y_traf_train)


    # 3. Do the same for the Traffic model later in the script
    preds_traf = model_traf.predict(X_traf_test_scaled)
    print("\n--- Traffic Congestion Classification Report ---")
    print(classification_report(y_traf_test, preds_traf))


    # Evaluate
    preds_traf = model_traf.predict(X_traf_test_scaled)
    traf_score = accuracy_score(y_traf_test, preds_traf)
    print(f" Traffic Model Accuracy: {traf_score * 100:.2f}%")

    # Save Model 2 artifacts
    joblib.dump(model_traf, os.path.join(models_dir, "traffic_model.pkl"))
    joblib.dump(scaler_traf, os.path.join(models_dir, "scaler_traffic.pkl"))
    joblib.dump(traffic_features, os.path.join(models_dir, "traffic_features.pkl"))
    print("   Saved: traffic_model.pkl, scaler_traffic.pkl, traffic_features.pkl")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print(" ALL MODELS TRAINED SUCCESSFULLY!")
    print(f" Models saved in: {models_dir}")
    print(f" Total Training Time: {time.time() - start_time:.2f} seconds")
    print("=" * 60)
    print(" Next Step: Run the API to serve these models!")


if __name__ == "__main__":
    train_models()
