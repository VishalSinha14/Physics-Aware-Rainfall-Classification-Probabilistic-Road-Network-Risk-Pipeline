import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

print("Loading monsoon temporal dataset...")

df = pd.read_csv("data/processed/era5_2022_monsoon_temporal.csv")

print("Total samples:", len(df))

# ---- Define Hazard Threshold ----
threshold = 10  # mm/hr
df["heavy_rain"] = (df["tp_mm"] >= threshold).astype(int)

print("\nClass distribution:")
print(df["heavy_rain"].value_counts())

features = [
    "t2m_c",
    "wind_speed",
    "sp",
    "swvl1",
    "rain_lag1",
    "rain_lag2",
    "rain_roll3",
    "rain_roll6"
]

X = df[features]
y = df["heavy_rain"]

print("\nSplitting dataset (stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("\nEvaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\nROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

# ---- Feature Importance ----
importances = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importances:")
print(importances.sort_values(ascending=False))