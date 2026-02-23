# Project Update -- Physics-Aware Rainfall Hazard & Probabilistic Road Risk Pipeline

## Overview

This project builds a probabilistic rainfall hazard modeling system
using ERA5 reanalysis data, machine learning, and bootstrap uncertainty
estimation. The goal is to integrate rainfall hazard probabilities with
road network vulnerability to compute risk scores.

------------------------------------------------------------------------

# Phase 1 -- Data Handling (Completed)

### Data Source

-   ERA5 hourly reanalysis (Single Levels)
-   Variables used:
    -   Total precipitation (tp)
    -   10m U/V wind components
    -   2m temperature
    -   Surface pressure
    -   Soil moisture (layer 1)

### Region

-   North: 26.5°
-   South: 18°
-   West: 104.5°
-   East: 117°

### Time Period

-   June 2022
-   July 2022
-   August 2022

### Processing Steps

-   Download monthly ERA5 data via CDS API
-   Extract ZIP-packaged NetCDF files
-   Merge `instant` + `accum` streams
-   Convert to DataFrame
-   Engineer features:
    -   tp_mm (mm)
    -   t2m_c (°C)
    -   wind_speed (m/s)

Final merged seasonal dataset: \~3.94 million rows

------------------------------------------------------------------------

# Phase 2 -- Temporal Feature Engineering (Completed)

Temporal features computed per grid cell:

-   rain_lag1
-   rain_lag2
-   rain_roll3
-   rain_roll6

Temporal continuity preserved across month boundaries.

Final seasonal temporal dataset: \~3.93 million rows

------------------------------------------------------------------------

# Phase 3 -- Rainfall Hazard Modeling (≥10 mm/hr)

## Hazard Definition

Heavy rainfall defined as:

    tp_mm ≥ 10 mm/hr

Class distribution: - Positive samples: 3,680 - Frequency: \~0.094%

## Model

Random Forest Classifier: - 300 trees - max_depth = 18 - class_weight =
balanced - Stratified train-test split

## Performance

Confusion Matrix: - Recall ≈ 0.96 - Precision ≈ 0.55

Metrics: - ROC-AUC ≈ 0.9998 - PR-AUC ≈ 0.886

Model shows strong discrimination for rare rainfall events.

Feature importance dominated by temporal rainfall persistence
features: - rain_roll3 - rain_roll6 - rain_lag1 - rain_lag2

------------------------------------------------------------------------

# Phase 3 (Extended) -- Bootstrap Uncertainty Ensemble

Bootstrap ensemble of 30 Random Forest models trained using resampled
datasets.

Outputs: - Mean rainfall probability - Standard deviation of predictions
(epistemic uncertainty)

Ensemble performance: - ROC-AUC ≈ 0.99985 - PR-AUC ≈ 0.886 - Mean
prediction std ≈ 0.00022 - Max prediction std ≈ 0.233

Uncertainty higher in borderline rainfall cases.

Saved outputs: - Ensemble prediction probabilities - Prediction
uncertainty (std)

------------------------------------------------------------------------

# Current Status

Completed: - Data ingestion - Seasonal dataset creation - Temporal
feature engineering - Rare-event hazard modeling - Probabilistic
uncertainty estimation

Next Phase: - Road Network Integration (OSM) - Vulnerability indexing -
Risk computation: Risk = Probability × Vulnerability - Risk uncertainty
propagation

------------------------------------------------------------------------

# Technical Stack

-   Python
-   Pandas
-   NumPy
-   Xarray
-   Scikit-learn
-   CDS API
-   ERA5 Reanalysis Data

------------------------------------------------------------------------

This repository now contains a working probabilistic rainfall hazard
modeling pipeline ready for spatial risk integration and dashboard
deployment.
