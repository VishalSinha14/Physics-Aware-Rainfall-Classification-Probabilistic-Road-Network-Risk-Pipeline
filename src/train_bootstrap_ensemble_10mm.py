import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

print("Loading monsoon temporal dataset...")

df = pd.read_csv("data/processed/era5_2022_monsoon_temporal.csv")

# ---- Define threshold ----
threshold = 10
df["heavy_rain"] = (df["tp_mm"] >= threshold).astype(int)

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

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training bootstrap ensemble...")

n_models = 30
all_probs = []

for i in range(n_models):

    print(f"Training model {i+1}/{n_models}")

    # Bootstrap sample from training data
    bootstrap_idx = np.random.choice(
        len(X_train),
        size=len(X_train),
        replace=True
    )

    X_boot = X_train.iloc[bootstrap_idx]
    y_boot = y_train.iloc[bootstrap_idx]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        class_weight="balanced",
        n_jobs=-1,
        random_state=i
    )

    model.fit(X_boot, y_boot)

    probs = model.predict_proba(X_test)[:, 1]
    all_probs.append(probs)

# Convert to array
all_probs = np.array(all_probs)

# Ensemble statistics
mean_prob = np.mean(all_probs, axis=0)
std_prob  = np.std(all_probs, axis=0)

print("\nEvaluating ensemble...")

roc_auc = roc_auc_score(y_test, mean_prob)
pr_auc  = average_precision_score(y_test, mean_prob)

print("Ensemble ROC-AUC:", roc_auc)
print("Ensemble PR-AUC:", pr_auc)

print("\nUncertainty statistics:")
print("Mean prediction std:", np.mean(std_prob))
print("Max prediction std :", np.max(std_prob))

# Save uncertainty outputs
output_df = X_test.copy()
output_df["true_label"] = y_test.values
output_df["mean_probability"] = mean_prob
output_df["uncertainty_std"] = std_prob

output_df.to_csv(
    "data/processed/monsoon_ensemble_predictions_10mm.csv",
    index=False
)

print("\nEnsemble predictions saved.")