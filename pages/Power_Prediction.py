import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Power Prediction",
    page_icon="⚡",
    layout="wide"
)

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

st.title("⚡ Solar Power Prediction")

st.markdown("""
Predict solar power generation using
weather and environmental conditions.
""")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("🌤 Enter Weather Parameters")

col1, col2 = st.columns(2)

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=28.0
    )

    humidity = st.number_input(
        "Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=55.0
    )

    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=50.0,
        value=5.0
    )

with col2:

    cloud_cover = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    radiation = st.number_input(
        "Solar Radiation (W/m²)",
        min_value=0.0,
        max_value=1500.0,
        value=800.0
    )

    pressure = st.number_input(
        "Pressure (hPa)",
        min_value=800.0,
        max_value=1200.0,
        value=1013.0
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🚀 Predict Power Generation"):

    input_dict = {

        "temperature_2_m_above_gnd": temperature,

        "relative_humidity_2_m_above_gnd": humidity,

        "wind_speed_10_m_above_gnd": wind_speed,

        "total_cloud_cover_sfc": cloud_cover,

        "shortwave_radiation_backwards_sfc": radiation,

        "pressure_reduced_to_msl": pressure
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in feature_names:

        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]

    st.success(
        f"⚡ Predicted Power Generation: "
        f"{prediction:.2f} kW"
    )

    # Performance Status

    if prediction > 300:

        st.success(
            "Excellent Solar Generation Expected ☀️"
        )

    elif prediction > 150:

        st.info(
            "Moderate Solar Generation Expected 🌤"
        )

    else:

        st.warning(
            "Low Solar Generation Expected ☁️"
        )

# --------------------------------------------------
# INFORMATION PANEL
# --------------------------------------------------

st.divider()

st.subheader("📌 Parameter Guidelines")

st.info("""
☀️ Higher Solar Radiation → Higher Power Generation

🌡 Moderate Temperature → Better Efficiency

☁️ Higher Cloud Cover → Lower Power Generation

💨 Wind Speed can help cool solar panels

💧 High Humidity may reduce efficiency
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Machine Learning Based Solar Power Forecasting System"
)
