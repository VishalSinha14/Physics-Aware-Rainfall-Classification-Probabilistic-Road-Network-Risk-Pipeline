# 🌧️ Rainfall to Road Risk Pipeline

## Getting Started Guide

------------------------------------------------------------------------

# 🎯 Project Goal

Build an end-to-end hazard → infrastructure risk forecasting system:

Satellite Rainfall Data (Time-Series)\
→ Temporal Feature Engineering\
→ Random Forest Classifier (Heavy Rain Probability)\
→ Bootstrap-Based Uncertainty Estimation\
→ Road Network Vulnerability Modeling\
→ Probabilistic Road Functionality Map\
→ Interactive Risk Dashboard

------------------------------------------------------------------------

# 📦 DATASETS REQUIRED

## 1️⃣ Rainfall Dataset (Hazard Data)

**Recommended:** ERA5 Reanalysis Rainfall Data\
- Variable: Total precipitation\
- Hourly resolution\
- Format: NetCDF (.nc)\
- Source: Copernicus Climate Data Store

Alternative: IMERG / TRMM Satellite Rainfall

------------------------------------------------------------------------

## 2️⃣ Elevation Data (Physics Component)

SRTM DEM\
- 30m resolution\
- Format: GeoTIFF\
- Source: USGS EarthExplorer

Used for: - Terrain influence\
- Elevation-based feature augmentation

------------------------------------------------------------------------

## 3️⃣ Road Network Data

OpenStreetMap (OSM)\
- Extract using `osmnx` Python library\
- Format: GraphML

------------------------------------------------------------------------

# 📁 FULL PROJECT FOLDER STRUCTURE

rainfall-risk-pipeline/

    │
    ├── data/
    │   ├── raw/
    │   │   ├── rainfall/
    │   │   ├── dem/
    │   │   └── roads/
    │   │
    │   ├── processed/
    │   │   ├── rainfall_features.csv
    │   │   ├── road_network.graphml
    │   │   └── merged_dataset.csv
    │
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_model_training.ipynb
    │
    ├── src/
    │   ├── data_loader.py
    │   ├── feature_engineering.py
    │   ├── rainfall_classifier.py
    │   ├── uncertainty.py
    │   ├── risk_model.py
    │   ├── network_model.py
    │   ├── evaluation_metrics.py
    │   ├── visualization.py
    │
    ├── app/
    │   ├── dashboard.py
    │
    ├── models/
    │   ├── random_forest.pkl
    │   ├── bootstrap_models/
    │
    ├── reports/
    │   ├── figures/
    │   ├── final_report.pdf
    │
    ├── requirements.txt
    ├── README.md
    └── main.py

------------------------------------------------------------------------

# 🔄 PROJECT FLOWCHART

ERA5 Rainfall Data\
→ Temporal Stacking (t-3, t-2, t-1)\
→ Add DEM Elevation\
→ Random Forest ML\
→ Bootstrap Ensemble\
→ Heavy Rain Probability\
→ Road Vulnerability Modeling\
→ Risk = Hazard × Vulnerability\
→ Interactive Risk Map

------------------------------------------------------------------------

# 🏗 ARCHITECTURE OVERVIEW

## 1️⃣ Hazard Modeling Layer

Feature Vector per Grid Cell:

-   rainfall_t-3\
-   rainfall_t-2\
-   rainfall_t-1\
-   current_rainfall\
-   elevation\
-   slope\
-   latitude\
-   longitude

Target: Heavy Rain (1/0)

------------------------------------------------------------------------

## 2️⃣ Random Forest Layer

Recommended Parameters:

-   n_estimators = 200\
-   max_depth tuned\
-   class_weight = balanced

Output: Probability of heavy rainfall

------------------------------------------------------------------------

## 3️⃣ Uncertainty Estimation

Bootstrap approach:

Train multiple RF models on resampled datasets.

Final Probability = Mean of predictions\
Uncertainty = Standard deviation of predictions

------------------------------------------------------------------------

## 4️⃣ Risk Modeling Layer

For each road segment:

Risk Score = Rainfall_Probability × Vulnerability_Index

Functionality = 1 − Risk Score

------------------------------------------------------------------------

# 🌐 FRONTEND + BACKEND

## Backend

Language: Python

Core Libraries: - pandas\
- numpy\
- scikit-learn\
- xarray\
- netCDF4\
- NetworkX\
- osmnx

Responsibilities: - Data preprocessing\
- Model inference\
- Risk calculation\
- Network updates

------------------------------------------------------------------------

## Frontend (Dashboard)

Framework: Streamlit

Visualization Tools: - Plotly\
- Folium

Dashboard Panels: 1. Rainfall probability map\
2. Road risk heatmap\
3. Evaluation metrics (POD, FAR, AUC)\
4. Uncertainty visualization

------------------------------------------------------------------------

# 📐 MATHEMATICAL COMPONENTS

## Random Forest Splitting

Gini Impurity:

G = 1 − Σ(p_i²)

------------------------------------------------------------------------

## Risk Formulation

Risk = Hazard × Vulnerability

------------------------------------------------------------------------

## Confusion Matrix Metrics

POD = TP / (TP + FN)\
FAR = FP / (TP + FP)

------------------------------------------------------------------------

## Uncertainty (Bootstrap)

σ = sqrt( (1/n) Σ (p_i − mean(p))² )

------------------------------------------------------------------------

# 📊 EVALUATION METRICS

Meteorological: - Probability of Detection (POD) - False Alarm Ratio
(FAR) - Critical Success Index (CSI) - ROC Curve - AUC

Infrastructure: - Average network functionality - Percentage of affected
critical roads

Uncertainty: - Variance of ensemble predictions

------------------------------------------------------------------------

# 🚀 Development Phases

Phase 1: Data Handling\
Phase 2: Model Training\
Phase 3: Uncertainty Modeling\
Phase 4: Road Network Integration\
Phase 5: Dashboard Development

------------------------------------------------------------------------

End of Getting Started Guide.
