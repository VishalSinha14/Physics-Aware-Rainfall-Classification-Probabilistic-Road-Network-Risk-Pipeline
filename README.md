# 🌧️ Physics-Aware Rainfall Classification & Probabilistic Road Network Risk Pipeline

An end-to-end machine learning pipeline that forecasts **road network risk** based on **ERA5 rainfall data**. The system classifies heavy rainfall events using bootstrap ensemble Random Forests, spatially joins hazard probabilities with OpenStreetMap road networks, and computes infrastructure vulnerability and risk scores — validated with research-grade scientific methods.

## 🏗️ Architecture

```
ERA5 NetCDF Data → Feature Engineering → Temporal Features (Lag, Rolling)
     → Random Forest Classification (10mm threshold)
     → Bootstrap Ensemble (30 models) → Probability + Uncertainty
     → Spatial Join with OSM Road Network
     → Vulnerability Index (Road Type + Betweenness Centrality)
     → Risk Score = Hazard × Vulnerability
     → Phase 6: Research-Grade Validation (9 components)
     → Interactive Streamlit Dashboard (6 pages)
```

## 📁 Project Structure

```
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment
├── Phase_6_Research_Grade_Validation_Plan.md
├── app/
│   └── dashboard.py                 # Streamlit dashboard (6 pages)
├── src/
│   ├── download_era5.py             # ERA5 data download (CDS API)
│   ├── download_era5_chunks.py      # Chunked ERA5 download
│   ├── download_era5_monthly.py     # Monthly ERA5 download
│   ├── download_single_month.py     # Single month download CLI
│   ├── process_era5.py              # ERA5 NetCDF → CSV processing
│   ├── process_june.py              # June data processing
│   ├── merge_monsoon_raw.py         # Merge Jun/Jul/Aug raw CSVs
│   ├── add_temporal_june.py         # Temporal feature engineering (June)
│   ├── add_temporal_monsoon.py      # Temporal features (full monsoon)
│   ├── check_rain_distribution.py   # Rain threshold analysis
│   ├── check_monsoon_distribution.py# Monsoon rain distribution
│   ├── train_rainfall_model.py      # Single RF classifier
│   ├── train_monsoon_10mm.py        # Monsoon RF (10mm threshold)
│   ├── train_bootstrap_ensemble_10mm.py  # Bootstrap ensemble (30 models)
│   ├── download_roads.py            # OSM road network download
│   ├── risk_model.py                # Road risk computation pipeline
│   ├── evaluation_metrics.py        # POD, FAR, CSI, infrastructure metrics
│   ├── visualization.py             # Plotting helpers (Folium + Plotly)
│   └── phase6_validation.py         # Phase 6: Research-grade validation
├── data/
│   ├── raw/                         # Raw ERA5 NetCDF files
│   └── processed/                   # Processed CSVs + GeoJSON outputs
├── models/
│   └── bootstrap_models/            # 30 trained RF model files (.pkl)
└── results/
    └── phase6/                      # Phase 6 validation plots + metrics
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate resilience_env
pip install streamlit-folium
```

### 2. Run the Full Pipeline
```bash
# Option A: Run everything via CLI
python main.py --all

# Option B: Run step by step
python src/train_bootstrap_ensemble_10mm.py   # Train 30 RF models
python src/download_roads.py                   # Download OSM roads
python src/risk_model.py                       # Compute risk scores

# Option C: Run Phase 6 validation
python src/phase6_validation.py                # ~30-40 min runtime

# Launch dashboard
python main.py --phase 5
# or: streamlit run app/dashboard.py
```

### 3. Check Prerequisites
```bash
python main.py --check
```

---

## 📊 Pipeline Phases

### Phase 1–2: Data Acquisition & Feature Engineering
- Downloads ERA5 reanalysis data (precipitation, temperature, wind, humidity)
- Engineers temporal features: lag-1, lag-2, lag-3 precipitation + rolling 6h sum
- Processes June–August 2022 monsoon season for Guangdong, China (3.9M samples)

### Phase 3: Hazard Classification
- Binary classification: **heavy rain ≥ 10 mm/hr**
- Bootstrap ensemble of **30 Random Forest** models (200 trees each)
- Produces **probabilistic predictions** with epistemic uncertainty
- Performance: **ROC-AUC ≈ 0.9999**, **PR-AUC ≈ 0.886**

### Phase 4: Road Network Risk
- Downloads OpenStreetMap road network (Guangzhou metro area, **153,472 segments**)
- **Spatial join**: nearest-neighbor matching of grid cells to road segments
- **Vulnerability index**: weighted combination of:
  - Road type vulnerability (motorway=0.1, residential=0.75, track=0.85)
  - Betweenness centrality from NetworkX graph analysis (69K nodes, 153K edges)
- **Risk = Hazard Probability × Vulnerability**
- **Functionality = 1 − Risk**

### Phase 5: Interactive Dashboard
Streamlit dashboard with **6 pages**:
- **📊 Overview**: Key metrics, risk/functionality distributions
- **🗺️ Risk Map**: Interactive Folium map with road risk overlay
- **📈 Metrics**: POD, FAR, CSI, infrastructure metrics
- **🔬 Uncertainty**: Ensemble prediction variance analysis
- **🧪 Phase 6 Validation**: All research-grade validation plots & metrics
- **📋 Data Explorer**: Raw data browser with CSV download

### Phase 6: Research-Grade Validation
Nine scientific validation components ensuring publication-readiness:

| # | Component | Key Result |
|---|-----------|------------|
| 1 | **Time-Based Train/Test Split** | ROC-AUC=0.9996, POD=0.88, CSI=0.58 (no data leakage) |
| 2 | **Reliability Diagram + Calibration** | Brier Score=0.000853, Isotonic regression applied |
| 3 | **Typhoon Stress Test** | **139x hazard amplification**, max prob=0.61 |
| 4 | **Dynamic Hazard Scaling** | At 3x rainfall: 3.4% roads high-risk, functionality=0.989 |
| 5 | **Threshold Sensitivity** | 5mm CSI=0.678, 10mm CSI=0.600, 20mm CSI=0.004 |
| 6 | **Multi-Threshold Fusion** | Graded severity (minor/moderate/major disruption) |
| 7 | **Spatial Cross-Validation** | ROC-AUC=0.9993 ± 0.0009 across 16 spatial blocks |
| 8 | **Monte Carlo Vulnerability** | Functionality 0.9999 ± 0.0000 (robust to ±20% perturbation) |
| 9 | **Ensemble Diversity** | Mixed RF+GBM achieves best Brier Score (0.00074) |

Additional: **CRPS**, **Brier decomposition** (Reliability–Resolution–Uncertainty), **Risk exceedance curves**

---

## 🛠️ Key Technologies

| Component | Technology |
|-----------|-----------|
| Data Source | ERA5 (ECMWF), OpenStreetMap |
| ML Framework | scikit-learn (Random Forest, Gradient Boosting) |
| Geospatial | GeoPandas, OSMnx, Shapely, Folium |
| Network Analysis | NetworkX (betweenness centrality) |
| Dashboard | Streamlit, Plotly, Folium |
| Data Processing | Pandas, NumPy, xarray |
| Calibration | Isotonic Regression (sklearn) |

## 📈 Evaluation Metrics

### Meteorological
- **POD** (Probability of Detection / Recall)
- **FAR** (False Alarm Ratio)
- **CSI** (Critical Success Index / Threat Score)
- **ROC-AUC** and **PR-AUC**

### Probabilistic
- **Brier Score** and decomposition (Reliability, Resolution, Uncertainty)
- **CRPS** (Continuous Ranked Probability Score)
- **Reliability diagrams** with calibration

### Infrastructure
- Average network functionality
- Critical road analysis (motorway/trunk/primary)
- Risk distribution (high/medium/low categories)
- Monte Carlo vulnerability robustness

### Uncertainty
- Epistemic uncertainty from bootstrap ensemble variance
- Multi-approach ensemble diversity comparison
- Risk exceedance probability curves

## 📍 Study Area

- **Region**: Guangdong Province, South China (Guangzhou metro area)
- **Coordinates**: 22.9°–23.4°N, 113.0°–113.5°E
- **Period**: June–August 2022 (East Asian Monsoon)
- **Data**: 3.9M hourly samples, ~153,000 road segments

## 🎯 Target Journals

- Structural Safety
- Reliability Engineering & System Safety
- Journal of Hydrometeorology
- Natural Hazards
- Environmental Research Letters

## 📜 License

This project is for academic/research purposes.