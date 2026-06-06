import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import preprocess_data
from src.visualizations import actual_vs_predicted

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return preprocess_data("data/spg.csv")

df = load_dataset()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/solar_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, features

model, feature_names = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📈 Machine Learning Model Performance")

st.markdown("""
Evaluate the solar power prediction model
using various performance metrics.
""")

# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

target_column = "generated_power_kw"

X = df.drop(columns=[target_column])

y = df[target_column]

X = pd.get_dummies(X)

for col in feature_names:
    if col not in X.columns:
        X[col] = 0

X = X[feature_names]

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

st.subheader("🎯 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "MAE",
        f"{mae:.3f}"
    )

with col2:
    st.metric(
        "MSE",
        f"{mse:.3f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"{rmse:.3f}"
    )

with col4:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

st.divider()

# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

st.subheader("🚀 Model Quality")

if r2 >= 0.90:

    st.success(
        "Excellent Model Performance"
    )

elif r2 >= 0.75:

    st.info(
        "Good Prediction Performance"
    )

elif r2 >= 0.50:

    st.warning(
        "Average Prediction Performance"
    )

else:

    st.error(
        "Model Requires Improvement"
    )

st.divider()

# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.subheader("📊 Actual vs Predicted Power")

st.plotly_chart(
    actual_vs_predicted(
        y_test.values,
        y_pred
    ),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("⭐ Feature Importance")

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
            model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

else:

    st.info(
        "Feature importance not available."
    )

st.divider()

# --------------------------------------------------
# ERROR ANALYSIS
# --------------------------------------------------

st.subheader("🔍 Prediction Error Analysis")

error_df = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": y_pred,

    "Error":
        y_test.values - y_pred

})

st.dataframe(
    error_df.head(20),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# PERFORMANCE SUMMARY
# --------------------------------------------------

st.subheader("📌 Summary")

st.markdown(f"""
### Model Evaluation Results

- MAE: **{mae:.3f}**
- RMSE: **{rmse:.3f}**
- R² Score: **{r2:.3f}**

### Interpretation

- Lower MAE and RMSE indicate better prediction accuracy.
- R² Score close to 1 indicates a strong predictive model.
- Solar radiation generally contributes most to power prediction.
- The model can be used for real-time solar power forecasting.

### Recommendation

Continue collecting weather and generation data
to improve model accuracy and reliability.
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Machine Learning Performance Dashboard | Solar Plant Efficiency Tracker"
)
