import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/spg.csv")

target = "generated_power_kw"

X = df.drop(columns=[target])
y = df[target]

X = pd.get_dummies(X)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "models/solar_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("Model Saved Successfully")
