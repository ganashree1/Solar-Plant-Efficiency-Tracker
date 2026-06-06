import streamlit as st

from src.preprocessing import preprocess_data
from src.analytics import (
    calculate_kpis,
    calculate_efficiency_score,
    dataset_summary
)

from src.visualizations import (
    power_generation_trend,
    radiation_vs_power,
    temperature_vs_power,
    power_distribution
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="☀️",
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

st.title("☀️ Solar Plant Executive Dashboard")

st.markdown("""
High-level overview of solar power generation,
weather impact, and plant efficiency.
""")

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

kpis = calculate_kpis(df)

efficiency_score = calculate_efficiency_score(df)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "⚡ Total Power",
        f"{kpis['Total Power']:,.0f} kW"
    )

with col2:
    st.metric(
        "📈 Average Power",
        f"{kpis['Average Power']:.2f} kW"
    )

with col3:
    st.metric(
        "🔥 Maximum Power",
        f"{kpis['Maximum Power']:.2f} kW"
    )

with col4:
    st.metric(
        "🌡 Avg Temp",
        f"{kpis['Average Temperature']:.2f} °C"
    )

with col5:
    st.metric(
        "🚀 Efficiency",
        f"{efficiency_score:.2f}%"
    )

st.divider()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("📋 Dataset Overview")

summary = dataset_summary(df)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", summary["Rows"])
c2.metric("Columns", summary["Columns"])
c3.metric("Missing Values", summary["Missing Values"])
c4.metric("Duplicates", summary["Duplicates"])

st.divider()

# --------------------------------------------------
# POWER TREND
# --------------------------------------------------

st.subheader("⚡ Power Generation Trend")

st.plotly_chart(
    power_generation_trend(df),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# WEATHER IMPACT ANALYSIS
# --------------------------------------------------

st.subheader("🌤 Weather Impact Analysis")

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        radiation_vs_power(df),
        use_container_width=True
    )

with col2:

    st.plotly_chart(
        temperature_vs_power(df),
        use_container_width=True
    )

st.divider()

# --------------------------------------------------
# POWER DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Generated Power Distribution")

st.plotly_chart(
    power_distribution(df),
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# PERFORMANCE INDICATOR
# --------------------------------------------------

st.subheader("🏭 Plant Performance Status")

if efficiency_score >= 80:

    st.success(
        "Excellent Performance - Plant is operating efficiently."
    )

elif efficiency_score >= 60:

    st.info(
        "Good Performance - Minor improvements possible."
    )

elif efficiency_score >= 40:

    st.warning(
        "Average Performance - Optimization recommended."
    )

else:

    st.error(
        "Low Performance - Immediate investigation required."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "AI Powered Solar Plant Efficiency Tracker | Streamlit Dashboard"
)
