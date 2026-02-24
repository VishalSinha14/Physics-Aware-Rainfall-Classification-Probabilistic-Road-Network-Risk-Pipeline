# Physics-Aware Rainfall Classification & Probabilistic Road Network Risk Pipeline

An end-to-end machine learning pipeline that forecasts **road network disruption risk** from ERA5 rainfall data. Uses a 30-model bootstrap Random Forest ensemble, spatial hazard-to-road mapping, and a full suite of research-grade validation including lead-time forecasting and baseline comparison.

**Study area:** Guangzhou metro area, Guangdong, China  
**Period:** June–August 2022 (East Asian Monsoon)  
**Hazard threshold:** ≥ 10 mm/hr (heavy rain)

---

## Pipeline Overview

```
ERA5 NetCDF  →  Feature Engineering (lag, rolling)
             →  Bootstrap RF Ensemble (30 models, 10mm threshold)
             →  Probability + Uncertainty per grid cell
             →  Spatial join → OSM Road Network (153K segments)
             →  Vulnerability Index (road type + betweenness centrality)
             →  Risk Score = Hazard × Vulnerability
             →  Phase 6: 9-component research validation
             →  Phase 7: LR baseline, lead-time forecasting, PR/Brier/reliability
             →  Streamlit Dashboard (8 pages)
```

---

## How to Reproduce

### 1. Clone and create environment

```bash
git clone https://github.com/VishalSinha14/Physics-Aware-Rainfall-Classification-Probabilistic-Road-Network-Risk-Pipeline
cd Physics-Aware-Rainfall-Classification-Probabilistic-Road-Network-Risk-Pipeline

conda env create -f environment.yml
conda activate resilience_env
pip install streamlit-folium
```

### 2. Download ERA5 data

You need a free [CDS API key](https://cds.climate.copernicus.eu/). Set it up at `~/.cdsapirc`, then:

```bash
python src/download_era5_monthly.py   # downloads Jun–Aug 2022 NetCDF files
```

> The raw `.nc` files are ~800MB and not included in this repo (see `.gitignore`).

### 3. Process ERA5 data

```bash
python src/process_era5.py            # converts NetCDF → CSV per month
python src/merge_monsoon_raw.py       # merges Jun/Jul/Aug CSVs
python src/add_temporal_monsoon.py    # adds lag/rolling features (~3.9M rows)
```

### 4. Download road network

```bash
python src/download_roads.py          # downloads OSM roads via OSMnx
```

### 5. Train the ensemble

```bash
python src/train_bootstrap_ensemble_10mm.py   # ~20–40 min on CPU
```

This trains 30 bootstrap RF models and saves them to `models/bootstrap_models/`.

### 6. Run the risk model

```bash
python src/risk_model.py              # spatial join + vulnerability + risk scores
```

Outputs `data/processed/road_risk_scores.geojson`.

### 7. Run Phase 6 validation

```bash
python src/phase6_validation.py       # ~30 min — 9 validation components
```

Outputs to `results/phase6/`.

### 8. Run Phase 7 research upgrades

Run these in order — each step depends on the previous one:

```bash
# Class imbalance analysis
python src/scientific/imbalance_analysis.py

# Logistic Regression baseline (takes ~15 min on 2.6M rows)
# NOTE: uses saga solver — do not change to lbfgs (crashes on Windows with large data)
python src/baselines/train_logistic_regression.py

# Model comparison table (LR vs Ensemble RF)
python src/baselines/model_comparison.py

# Lead-time label engineering (creates 0h/30m/60m CSVs)
python src/leadtime/engineer_lead_labels.py

# Train lead-time RF models (10 models × 3 lead times, ~60 min)
python src/leadtime/train_leadtime_models.py

# Generate degradation curve plot
python src/leadtime/evaluate_leadtime.py

# Scientific plots (Brier, PR curves, reliability diagrams)
python src/scientific/brier_analysis.py
python src/scientific/reliability_curves.py
python src/scientific/pr_curves.py
```

### 9. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

---

## Project Structure

```
├── src/
│   ├── download_era5_monthly.py        # ERA5 data download
│   ├── process_era5.py                 # NetCDF → CSV
│   ├── merge_monsoon_raw.py            # merge monthly CSVs
│   ├── add_temporal_monsoon.py         # lag/rolling feature engineering
│   ├── train_bootstrap_ensemble_10mm.py# 30-model RF ensemble
│   ├── download_roads.py               # OSM road download
│   ├── risk_model.py                   # hazard × vulnerability → risk
│   ├── evaluation_metrics.py           # POD, FAR, CSI, Brier
│   ├── visualization.py                # Folium + Plotly helpers
│   ├── phase6_validation.py            # Phase 6: 9-component validation
│   ├── baselines/
│   │   ├── train_logistic_regression.py# LR baseline model
│   │   └── model_comparison.py         # LR vs RF comparison table
│   ├── leadtime/
│   │   ├── engineer_lead_labels.py     # shift labels by 1/2 ERA5 steps
│   │   ├── train_leadtime_models.py    # train RF at 0h, 30m, 60m
│   │   └── evaluate_leadtime.py        # degradation curve plot
│   └── scientific/
│       ├── imbalance_analysis.py       # class imbalance + why accuracy fails
│       ├── brier_analysis.py           # Brier Score comparison
│       ├── reliability_curves.py       # calibration diagrams all models
│       └── pr_curves.py                # Precision-Recall curves
├── app/
│   └── dashboard.py                    # Streamlit dashboard (8 pages)
├── data/
│   ├── raw/                            # ERA5 NetCDF (not in repo, ~800MB)
│   └── processed/                      # CSVs, GeoJSON outputs
├── models/
│   ├── bootstrap_models/               # 30 × Phase 3 RF models (.pkl)
│   └── leadtime/                       # 30 × lead-time RF models (.pkl)
├── results/
│   ├── phase6/                         # 9 validation plots + metrics
│   ├── baselines/                      # LR results + comparison table
│   ├── leadtime/                       # degradation curve + metrics
│   └── scientific/                     # Brier, PR, reliability plots
├── environment.yml
├── requirements.txt
└── main.py                             # CLI entry point
```

> **Note:** `.pkl` model files and large CSVs (>50MB) are excluded from Git. See `.gitignore`.

---

## Phases & Key Results

### Phase 1–2: Data & Feature Engineering
- ERA5 variables: precipitation, 2m temperature, wind speed, soil moisture, surface pressure
- Temporal features: lag-1, lag-2, lag-3 precipitation, 3h and 6h rolling sums
- Dataset: 3.9M hourly samples across 1,785 grid points

### Phase 3: Hazard Classification
- **30 bootstrap RF models**, 200 trees each, `class_weight='balanced'`
- Test ROC-AUC ≈ 0.9999, PR-AUC ≈ 0.886

### Phase 4: Road Network Risk
- 153,472 OSM road segments, 69K network nodes
- Vulnerability = road type weight × betweenness centrality
- Risk = Hazard probability × Vulnerability

### Phase 5: Dashboard
8-page Streamlit dashboard: Overview, Risk Map, Metrics, Uncertainty, Phase 6 Validation, Model Comparison, Lead-Time, Data Explorer.

### Phase 6: Research-Grade Validation

| Component | Key Result |
|---|---|
| Time-based split (no leakage) | ROC-AUC=0.9996, CSI=0.58 |
| Calibration + reliability diagram | Brier Score=0.000853 |
| Typhoon stress test | 139× hazard amplification |
| Dynamic hazard scaling | 3.4% roads at high risk under 3× rainfall |
| Threshold sensitivity (5/10/20 mm) | CSI: 0.678 / 0.600 / 0.004 |
| Multi-threshold severity fusion | Three-tier graded risk output |
| Spatial cross-validation (16 blocks) | AUC=0.9993 ± 0.0009 |
| Monte Carlo vulnerability | Functionality stable under ±20% perturbation |
| Ensemble diversity (RF vs GBM) | Mixed RF+GBM best Brier = 0.00074 |

### Phase 7: Research Upgrades

**Baseline comparison**
| Model | POD | CSI | ROC-AUC | Brier |
|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.385 | 0.9989 | 0.00369 |
| Ensemble RF (30 models) | 0.931 | 0.592 | 0.9995 | 0.00113 |

The ensemble RF has 54% better CSI and 69% lower Brier than LR — justifying the added complexity.

**Lead-time forecasting** (converts system from nowcast to early-warning)

| Lead Time | POD | CSI | ROC-AUC |
|---|---|---|---|
| 0h (nowcast) | 0.931 | 0.592 | 0.9995 |
| +30 min | 0.731 | 0.279 | 0.9921 |
| +60 min | 0.599 | 0.180 | 0.9722 |

**Why POD and CSI, not accuracy?**  
With a 0.09% positive rate (heavy rain events), a model that always predicts "no rain" gets 99.91% accuracy — but zero skill. We use POD, FAR, CSI, PR-AUC, and Brier Score instead.

---

## Technologies

| Layer | Tools |
|---|---|
| Data | ERA5 (CDS API), OpenStreetMap (OSMnx) |
| ML | scikit-learn (RF, LR, GBM, Isotonic calibration) |
| Geospatial | GeoPandas, Shapely, Folium |
| Network analysis | NetworkX |
| Dashboard | Streamlit, Plotly |
| Environment | Conda (Python 3.10) |

---

## License

Academic/research use only.