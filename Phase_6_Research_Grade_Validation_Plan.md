# 🌧️ Phase 6: Research-Grade Validation & Enhancement Plan

## Physics-Aware Rainfall → Probabilistic Road Network Risk Pipeline

------------------------------------------------------------------------

# 🎯 Objective

Elevate the current prototype into a **research-accurate,
publication-ready, operational-grade forecasting system** aligned with
best practices in:

-   Probabilistic meteorological forecasting
-   Infrastructure resilience modeling
-   Uncertainty quantification
-   Real-time risk prediction

------------------------------------------------------------------------

# 🔬 Core Philosophy

We move from:

> "Working System"

To:

> "Scientifically Defensible, Stress-Tested, Generalizable Risk Engine"

This phase ensures: - No data leakage - Proper generalization - Reliable
probabilities - Realistic hazard stress scenarios - Spatial robustness -
Infrastructure-level interpretability

------------------------------------------------------------------------

# 🧭 Phase 6 Upgrade Roadmap

------------------------------------------------------------------------

## 1️⃣ Proper Time-Based Train/Test Split

### 🎯 Why

Meteorological forecasting must respect temporal causality. Random
splits inflate AUC and cause leakage.

### 📌 Plan

-   Split by time blocks:
    -   Train: June--July 2022
    -   Validation: Early August 2022
    -   Test: Late August 2022
-   Ensure lag features use only past data
-   Retrain bootstrap ensemble using time-aware split
-   Recompute ROC-AUC, PR-AUC, POD, FAR, CSI

### 📊 Expected Outcome

More realistic AUC (\~0.85--0.95 range).

------------------------------------------------------------------------

## 2️⃣ Reliability Diagram (Probability Calibration)

### 🎯 Why

High AUC does not guarantee calibrated probabilities.

### 📌 Plan

-   Bin predicted probabilities (0--1) into deciles
-   Compute observed frequency per bin
-   Plot reliability curve
-   Compute Brier Score
-   Apply:
    -   Isotonic Regression
    -   Platt Scaling
-   Re-evaluate calibration after correction

### 📊 Expected Outcome

Hazard probabilities become statistically meaningful.

------------------------------------------------------------------------

## 3️⃣ Typhoon Case Injection (Extreme Event Stress Test)

### 🎯 Why

Model currently under-stressed.

### 📌 Plan

-   Identify historical typhoon rainfall event in South China
-   Inject high-intensity rainfall sequences
-   Run hazard model inference
-   Compute new risk distribution
-   Compare baseline vs extreme case

### 📊 Expected Outcome

Non-zero high-risk road clusters Realistic functionality degradation
patterns

------------------------------------------------------------------------

## 4️⃣ Dynamic Hazard Scaling Experiment

### 🎯 Why

Test risk engine stability under amplified climate signals.

### 📌 Plan

For rainfall input: - Multiply by factors: 1.0x, 1.5x, 2.0x, 3.0x -
Recompute hazard probabilities - Propagate to risk model - Measure: -
High-risk percentage - Network functionality decline - Critical road
vulnerability

### 📊 Output

Risk elasticity curves: Risk vs Rainfall Intensity

------------------------------------------------------------------------

## 5️⃣ Sensitivity of Risk to Threshold

### 🎯 Why

10mm/hr threshold may suppress hazard variability.

### 📌 Plan

Train three classifiers: - 5mm/hr - 10mm/hr - 20mm/hr

For each: - Evaluate metrics - Compare hazard probability
distributions - Compare infrastructure risk outputs

### 📊 Expected Outcome

Understand threshold-induced bias in risk estimation.

------------------------------------------------------------------------

## 6️⃣ Compare 5mm, 10mm, 20mm Multi-Threshold Fusion

### 🎯 Advanced Upgrade

-   Build multi-output classifier predicting multiple thresholds
-   Model hazard severity levels
-   Convert risk from binary to graded severity risk
-   Generate:
    -   Minor disruption probability
    -   Major disruption probability

------------------------------------------------------------------------

## 7️⃣ Spatial Cross-Validation

### 🎯 Why

Prevent spatial overfitting.

### 📌 Plan

-   Divide grid into spatial blocks
-   Perform leave-one-region-out validation
-   Evaluate generalization across geography
-   Compare metrics to temporal validation

### 📊 Expected Outcome

Spatial robustness assessment.

------------------------------------------------------------------------

# 📈 Infrastructure Validation Layer

After hazard validation:

### 8️⃣ Functionality Validation

-   Compare predicted functionality to historical traffic disruption
    data (if available)
-   Validate centrality-weighted vulnerability formulation
-   Perform Monte Carlo perturbation of vulnerability weights

------------------------------------------------------------------------

# 🔎 Uncertainty Upgrade

### 9️⃣ Ensemble Diversity Improvement

Current bootstrap variance too small.

Enhancements: - Random feature subsets - Different RF hyperparameters
per bootstrap - Add Gradient Boosting ensemble - Compare epistemic
spread

------------------------------------------------------------------------

# 📊 Additional Research Metrics

-   Brier Score
-   Continuous Ranked Probability Score (CRPS)
-   Reliability--Resolution--Uncertainty decomposition
-   Risk exceedance probability curves
-   Network robustness index under stress

------------------------------------------------------------------------

# 🧪 Final Research Deliverables

After Phase 6:

✔ Time-aware validated hazard model\
✔ Calibrated probabilistic forecasts\
✔ Stress-tested typhoon scenario outputs\
✔ Climate scaling experiment curves\
✔ Threshold sensitivity analysis\
✔ Spatial generalization results\
✔ Improved epistemic uncertainty modeling\
✔ Scientifically defensible infrastructure risk metrics

------------------------------------------------------------------------

# 🚀 Path Toward Publication

Target Journals:

-   Structural Safety
-   Reliability Engineering & System Safety
-   Journal of Hydrometeorology
-   Natural Hazards
-   Environmental Research Letters

------------------------------------------------------------------------

# 🏁 Final Goal

Transform system into:

> Real-time regional-scale probabilistic infrastructure risk forecasting
> engine under evolving rainfall hazards

With:

-   Calibration
-   Generalization
-   Stress realism
-   Infrastructure interpretability
-   Quantified uncertainty
-   Policy-ready outputs

------------------------------------------------------------------------

**End of Phase 6 Plan**
