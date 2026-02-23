import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load temporal dataset
df = pd.read_csv("data/processed/era5_20220601_temporal.csv")

# ---- Define Target ----
df['heavy_rain'] = (df['tp_mm'] >= 5).astype(int)

# ---- Features ----
features = [
    't2m_c',
    'wind_speed',
    'sp',
    'swvl1',
    'rain_lag1',
    'rain_lag2',
    'rain_roll3',
    'rain_roll6'
]

X = df[features]
y = df['heavy_rain']

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Train Model ----
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))