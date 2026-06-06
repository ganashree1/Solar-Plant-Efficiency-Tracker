import streamlit as st

from src.preprocessing import preprocess_data
from src.visualizations import (
    temperature_vs_power,
    humidity_vs_power,
    wind_speed_vs_power,
    cloud_cover_vs_power,
    radiation_vs_power
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Weather Analysis",
    page_icon="🌤",
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
# HEADER
# --------------------------------------------------

st.title("🌤 Weather Impact Analysis")

st.markdown("""
Analyze how weather parameters affect
solar power generation.
""")

# --------------------------------------------------
# WEATHER SUMMARY
# --------------------------------------------------

st.subheader("📊 Weather Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌡 Avg Temperature",
        f"{df['temperature_2_m_above_gnd'].mean():.2f} °C"
    )

with col2:
    st.metric(
        "💧 Avg Humidity",
        f"{df['relative_humidity_2_m_above_gnd'].mean():.2f}%"
    )

with col3:
    st.metric(
        "💨 Avg Wind Speed",
        f"{df['wind_speed_10_m_above_gnd'].mean():.2f} m/s"
    )

with col4:
    st.metric(
        "☁ Avg Cloud Cover",
        f"{df['total_cloud_cover_sfc'].mean():.2f}%"
    )

st.divider()

# --------------------------------------------------
# RADIATION ANALYSIS
# --------------------------------------------------

st.subheader("☀ Solar Radiation Impact")

st.plotly_chart(
    radiation_vs_power(df),
    use_container_width=True
)

st.info("""
Solar radiation is the most important
factor affecting solar power generation.
Higher radiation generally produces
higher power output.
""")

st.divider()

# --------------------------------------------------
# TEMPERATURE ANALYSIS
# --------------------------------------------------

st.subheader("🌡 Temperature Impact")

st.plotly_chart(
    temperature_vs_power(df),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# HUMIDITY ANALYSIS
# --------------------------------------------------

st.subheader("💧 Humidity Impact")

st.plotly_chart(
    humidity_vs_power(df),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# WIND SPEED ANALYSIS
# --------------------------------------------------

st.subheader("💨 Wind Speed Impact")

st.plotly_chart(
    wind_speed_vs_power(df),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# CLOUD COVER ANALYSIS
# --------------------------------------------------

st.subheader("☁ Cloud Cover Impact")

st.plotly_chart(
    cloud_cover_vs_power(df),
    use_container_width=True
)

st.warning("""
Higher cloud cover reduces the amount
of sunlight reaching solar panels,
which lowers power generation.
""")

st.divider()

# --------------------------------------------------
# WEATHER INSIGHTS
# --------------------------------------------------

st.subheader("🤖 Weather Insights")

temperature_corr = df[
    "temperature_2_m_above_gnd"
].corr(
    df["generated_power_kw"]
)

humidity_corr = df[
    "relative_humidity_2_m_above_gnd"
].corr(
    df["generated_power_kw"]
)

wind_corr = df[
    "wind_speed_10_m_above_gnd"
].corr(
    df["generated_power_kw"]
)

cloud_corr = df[
    "total_cloud_cover_sfc"
].corr(
    df["generated_power_kw"]
)

radiation_corr = df[
    "shortwave_radiation_backwards_sfc"
].corr(
    df["generated_power_kw"]
)

st.success(
    f"☀ Radiation Correlation: {radiation_corr:.2f}"
)

st.info(
    f"🌡 Temperature Correlation: {temperature_corr:.2f}"
)

st.info(
    f"💧 Humidity Correlation: {humidity_corr:.2f}"
)

st.info(
    f"💨 Wind Speed Correlation: {wind_corr:.2f}"
)

st.warning(
    f"☁ Cloud Cover Correlation: {cloud_corr:.2f}"
)

st.divider()

# --------------------------------------------------
# CONCLUSION
# --------------------------------------------------

st.subheader("📌 Key Findings")

st.markdown("""
### Main Observations

- ☀ Solar radiation has the strongest impact on power generation.
- 🌡 Moderate temperatures improve panel performance.
- ☁ High cloud cover reduces energy production.
- 💨 Wind can improve efficiency by cooling panels.
- 💧 Humidity may slightly affect performance.

### Recommendation

Monitor weather forecasts and radiation levels
to maximize solar plant efficiency and improve
power generation forecasting.
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Weather Analytics Module | Solar Plant Efficiency Tracker"
)
