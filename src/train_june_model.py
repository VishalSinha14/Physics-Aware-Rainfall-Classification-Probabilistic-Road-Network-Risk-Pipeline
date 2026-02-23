import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("Loading June temporal dataset...")

df = pd.read_csv("data/processed/era5_2022_06_temporal.csv")

# Define threshold
threshold = 5
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

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluating...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))