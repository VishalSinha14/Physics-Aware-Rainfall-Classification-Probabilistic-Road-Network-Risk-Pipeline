# Research Upgrade Plan -- Scientific Rigor & Publication Readiness

## Project: Physics-Aware Rainfall Classification & Probabilistic Road Risk System

------------------------------------------------------------------------

# OVERVIEW

This document outlines the structured execution plan to upgrade the
current system from a strong engineering project to a publication-ready
scientific framework.

The upgrades focus on:

1.  Eliminating misleading accuracy emphasis
2.  Adding proper probabilistic metrics
3.  Introducing a Logistic Regression baseline
4.  Implementing lead-time forecasting (+30 min, +1 hour)
5.  Adding scientific solidification components (Brier Score,
    reliability curve, PR curve, imbalance analysis)

Each section below explains:

-   Why it is necessary
-   What exactly will be implemented
-   How it will be implemented (technical steps)
-   Expected outputs
-   Validation criteria

------------------------------------------------------------------------

# 1. REMOVE MISLEADING ACCURACY EMPHASIS

## Problem

Heavy rainfall is rare. Accuracy is inflated due to dominant True
Negatives (TN).

Accuracy = (TP + TN) / Total

Since TN is very large, accuracy becomes artificially high (99.96%).

Serious reviewers will reject a paper that highlights accuracy for
rare-event detection.

## Strategy

We will:

-   Remove accuracy from main results table
-   Move accuracy to appendix only
-   Emphasize rare-event metrics

## Primary Metrics to Report

-   POD (Probability of Detection)
-   FAR (False Alarm Ratio)
-   CSI (Critical Success Index)
-   AUC (Area Under ROC Curve)
-   Brier Score (probabilistic calibration)

## Implementation Steps

1.  Modify evaluation script:
    -   Remove accuracy from primary print statements
    -   Keep POD, FAR, CSI, AUC
2.  Update dashboard:
    -   Replace "Accuracy" display card with:
        -   POD
        -   FAR
        -   CSI
        -   AUC
3.  Update documentation:
    -   Add explanation section: "Accuracy is not meaningful under
        severe class imbalance."
4.  Add confusion matrix normalization view:
    -   Show class-wise recall and precision

## Expected Outcome

-   Scientifically defensible evaluation
-   Reviewer-proof metric reporting

------------------------------------------------------------------------

# 2. ADD LOGISTIC REGRESSION BASELINE

## Why This Is Necessary

Currently only ensemble RF-based models exist.

To prove improvement, we need a simple statistical baseline.

Logistic regression is: - Interpretable - Standard in meteorology -
Expected by reviewers

## Experimental Setup

We will compare:

Model A: Logistic Regression Model B: Single Random Forest Model C:
Bootstrap Ensemble RF (current best)

All trained with identical: - Temporal split - Spatial
cross-validation - Threshold definition

## Technical Implementation

### Step 1: Feature Preparation

Use same feature set as ensemble model: - Rainfall features - Wind
speed - Soil moisture - Temperature - Lag features (1--3 hours)

Standardize features using StandardScaler.

### Step 2: Train Logistic Regression

Use: - Class_weight='balanced' - L2 regularization - Hyperparameter
tuning (C parameter)

### Step 3: Evaluation

Compute:

-   POD
-   FAR
-   CSI
-   AUC
-   Brier Score
-   Reliability curve

### Step 4: Statistical Comparison

Perform:

-   McNemar test for classification difference
-   AUC comparison

## Output Tables

Create comparison table:

  Model   POD   FAR   CSI   AUC   Brier
  ------- ----- ----- ----- ----- -------

## Expected Scientific Outcome

If ensemble significantly outperforms logistic regression, we prove
nonlinear feature learning adds value.

------------------------------------------------------------------------

# 3. LEAD-TIME STUDY (+30 min, +1 hour)

## Current Limitation

System is 0-hour nowcasting.

This limits scientific contribution.

## Goal

Extend model to predict:

-   t + 30 minutes
-   t + 60 minutes

## Data Engineering Plan

### Step 1: Label Shifting

For each timestamp t:

Create labels: - Y_30 = heavy_rain at t+30 - Y_60 = heavy_rain at t+60

Align features from time t with future labels.

### Step 2: Remove Leakage

Ensure: - Only past data used for prediction - No future weather
features included

### Step 3: Train Separate Models

Train three models:

-   Model_0h
-   Model_30m
-   Model_60m

### Step 4: Evaluate Degradation Curve

Plot:

Lead Time vs POD Lead Time vs CSI Lead Time vs AUC

## Expected Behavior

Performance should decrease smoothly as lead-time increases.

If collapse occurs → feature set insufficient.

## Scientific Contribution

This converts project from detection system to predictive hazard
early-warning framework.

------------------------------------------------------------------------

# 4. SCIENTIFIC SOLIDIFICATION PHASE

------------------------------------------------------------------------

## 4A. ADD BRIER SCORE

### Why

Measures probabilistic calibration quality.

Formula:

Brier = mean((p_i - y_i)\^2)

Lower is better.

### Implementation

Use sklearn.metrics.brier_score_loss.

Compute for: - 0h - 30m - 60m

Add to model comparison table.

------------------------------------------------------------------------

## 4B. RELIABILITY CURVE

### Why

Tests probability honesty.

Plot: - Predicted probability bins (0--1) - Observed frequency

### Implementation

1.  Bin predictions (10 bins)
2.  Compute empirical frequency
3.  Plot diagonal reference line

Add plot to:

-   Paper
-   Dashboard

------------------------------------------------------------------------

## 4C. PRECISION-RECALL CURVE

### Why

PR curve is better than ROC for imbalanced data.

### Implementation

Use sklearn precision_recall_curve.

Compute:

-   PR AUC
-   Optimal threshold analysis

Add PR plot to validation folder.

------------------------------------------------------------------------

## 4D. CLASS IMBALANCE DISCUSSION

Add paper section:

### 1. Class Distribution

Report:

-   \% heavy rain
-   \% non-heavy rain

### 2. Why Accuracy Fails

Explain mathematically.

### 3. Mitigation Strategy

-   Balanced class weights
-   Ensemble modeling
-   Metric selection shift

------------------------------------------------------------------------

# EXECUTION TIMELINE

Week 1: - Remove accuracy emphasis - Add Brier Score - Add PR curve

Week 2: - Implement Logistic Regression baseline - Produce comparison
table

Week 3: - Engineer lead-time labels - Train 30m and 60m models

Week 4: - Full evaluation - Update documentation - Prepare
publication-ready figures

------------------------------------------------------------------------

# FINAL TARGET STATE

After completing all steps:

System will include:

-   Proper rare-event evaluation
-   Probabilistic calibration metrics
-   Baseline comparison
-   Lead-time forecasting capability
-   Scientifically defensible documentation

This will elevate the project to:

Regional-scale probabilistic hazard early-warning framework suitable for
journal submission in infrastructure resilience or climate risk modeling
domains.

------------------------------------------------------------------------

END OF PLAN
