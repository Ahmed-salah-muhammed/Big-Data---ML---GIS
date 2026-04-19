# 🚀 Real-Time Accident & Traffic Prediction System (Big Data + ML + GIS)

## 📌 Project Overview

This project aims to build a **real-time accident and traffic prediction system** using large-scale accident data. The system leverages **Big Data technologies, Machine Learning models, and GIS visualization** to predict accident probability and estimate traffic congestion levels at any given location and time.

---

## 🎯 Objectives

* Predict the probability of traffic accidents based on spatial, temporal, and environmental factors
* Estimate traffic congestion levels using accident density as a proxy
* Integrate predictions into an interactive GIS dashboard
* Simulate real-time predictions via API endpoints

---

## 🧠 Key Concepts

* Big Data Processing
* Machine Learning (Classification)
* Spatial Analysis (GIS)
* Real-Time API Simulation

---

## 📊 Dataset

* US Accidents Dataset (2016–2023)
* Contains over 500,000 records including:

  * Location (Latitude, Longitude)
  * Time (Start Time)
  * Weather conditions
  * Visibility
  * Temperature
  * Severity

---

## 🧱 System Architecture

Data flows through the following pipeline:

Data Source (CSV)
→ Data Processing (PySpark)
→ Feature Engineering
→ Traffic Label Generation
→ Machine Learning Model
→ API (FastAPI)
→ GIS Dashboard Visualization

---

## ⚙️ Technologies Used

* PySpark for large-scale data processing
* Python (Pandas, NumPy) for data handling
* Scikit-learn for machine learning
* FastAPI for serving predictions
* GIS tools (ArcGIS / Leaflet) for visualization

---

## 🔄 Data Processing Steps

### 1. Data Cleaning

* Remove null and inconsistent values
* Select relevant columns
* Convert data types

### 2. Feature Engineering

* Extract time features (hour, day, rush hour)
* Encode weather conditions (Rain, Fog, Clear)
* Create spatial grid system using latitude and longitude

### 3. Traffic Labeling (Derived Feature)

Traffic congestion is not directly available in the dataset.
Therefore, it is derived using accident density:

* Group data by spatial grid and time
* Count number of accidents per grid per hour
* Assign traffic level:

  * High Traffic → if accident count exceeds threshold
  * Low Traffic → otherwise

---

## 🤖 Machine Learning

### Problem Type

* Classification

### Targets

* Accident occurrence probability
* Traffic congestion level

### Models Used

* Random Forest Classifier
* Logistic Regression (baseline)

---

## 🌐 API Design (FastAPI)

### Endpoint

POST /predict

### Input

* Latitude
* Longitude
* Temperature
* Visibility
* Hour
* Weather condition

### Output

* Accident Probability
* Risk Level (Low / Medium / High)
* Traffic Level (Low / High)

---

## 🗺️ GIS Integration

The system integrates predictions into a map-based dashboard:

* Display accident risk as color-coded markers
* Visualize traffic congestion using heatmaps
* Enable user interaction (click on map to get predictions)

---

## 🔥 Advanced Features

* Batch prediction for heatmap generation
* Real-time simulation via API
* Spatial indexing using grid system
* Interactive dashboard with filtering by time and weather

---

## 💡 Innovation

This project introduces a smart approach by:

* Using accident density as a proxy for traffic congestion
* Combining Big Data processing with GIS visualization
* Simulating real-time decision support systems for smart cities

---

## 🎯 Expected Outcomes

* Accurate prediction of accident-prone areas
* Identification of high-risk time periods
* Visual insights into traffic and safety patterns
* A complete end-to-end data pipeline from raw data to visualization

---

## 🏁 Conclusion

This project demonstrates how integrating Big Data, Machine Learning, and GIS can provide powerful insights for urban planning, traffic management, and road safety improvement.
