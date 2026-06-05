import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import load_data, clean_data
from src.analytics import calculate_kpis
from src.insights import generate_insights

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Solar Plant Efficiency Tracker",
    page_icon="☀️",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align:center;
}

.insight-box {
    background:#eef7ff;
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def get_data():
    df = load_data("data/spg.csv")
    df = clean_data(df)
    return df

df = get_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.image(
    "logo.png",
    width=150
)

st.sidebar.title("☀️ Solar Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Weather Analysis",
        "Correlation Analysis",
        "AI Insights"
    ]
)

# ---------------------------------------------------
# KPIs
# ---------------------------------------------------

kpis = calculate_kpis(df)

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

if page == "Dashboard":

    st.title("☀️ Solar Plant Efficiency Tracker")

    st.markdown(
        "Real-Time Analytics for Solar Power Generation"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "⚡ Total Power",
        f"{kpis['Total Power']:,.0f} kW"
    )

    col2.metric(
        "📈 Average Power",
        f"{kpis['Average Power']:.2f} kW"
    )

    col3.metric(
        "🔥 Maximum Power",
        f"{kpis['Maximum Power']:.2f} kW"
    )

    col4.metric(
        "🌡 Avg Temperature",
        f"{kpis['Average Temperature']:.2f}°C"
    )

    st.markdown("---")

    st.subheader("⚡ Power Generation Trend")

    fig = px.line(
        df,
        y="generated_power_kw",
        title="Generated Power Over Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("☀ Radiation vs Power")

        fig2 = px.scatter(
            df,
            x="shortwave_radiation_backwards_sfc",
            y="generated_power_kw",
            trendline="ols"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with col2:

        st.subheader("🌡 Temperature vs Power")

        fig3 = px.scatter(
            df,
            x="temperature_2_m_above_gnd",
            y="generated_power_kw",
            trendline="ols"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

# ---------------------------------------------------
# WEATHER ANALYSIS
# ---------------------------------------------------

elif page == "Weather Analysis":

    st.title("🌤 Weather Impact Analysis")

    weather_cols = [
        "temperature_2_m_above_gnd",
        "relative_humidity_2_m_above_gnd",
        "wind_speed_10_m_above_gnd",
        "total_cloud_cover_sfc"
    ]

    selected = st.selectbox(
        "Select Weather Parameter",
        weather_cols
    )

    fig = px.scatter(
        df,
        x=selected,
        y="generated_power_kw",
        color="generated_power_kw",
        title=f"{selected} vs Generated Power"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.histogram(
        df,
        x=selected,
        nbins=30,
        title=f"Distribution of {selected}"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ---------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------

elif page == "Correlation Analysis":

    st.title("📊 Correlation Analysis")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Top Correlations with Generated Power"
    )

    power_corr = (
        corr["generated_power_kw"]
        .sort_values(ascending=False)
    )

    st.dataframe(power_corr)

# ---------------------------------------------------
# AI INSIGHTS
# ---------------------------------------------------

elif page == "AI Insights":

    st.title("🤖 AI Generated Insights")

    insights = generate_insights(df)

    for insight in insights:

        st.markdown(
            f"""
            <div class="insight-box">
            ✅ {insight}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("📌 Summary")

    corr = df.corr(numeric_only=True)

    important_feature = (
        corr["generated_power_kw"]
        .drop("generated_power_kw")
        .abs()
        .idxmax()
    )

    st.success(
        f"Most influential feature affecting power generation: "
        f"{important_feature}"
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, Plotly & Machine Learning"
)
