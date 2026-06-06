import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def power_generation_trend(df):
    """
    Line chart for generated power
    """

    fig = px.line(
        df,
        y="generated_power_kw",
        title="Power Generation Trend"
    )

    fig.update_layout(
        xaxis_title="Records",
        yaxis_title="Generated Power (kW)",
        template="plotly_white"
    )

    return fig


def radiation_vs_power(df):
    """
    Solar Radiation vs Power
    """

    fig = px.scatter(
        df,
        x="shortwave_radiation_backwards_sfc",
        y="generated_power_kw",
        color="generated_power_kw",
        title="Solar Radiation vs Generated Power",
        trendline="ols"
    )

    return fig


def temperature_vs_power(df):
    """
    Temperature vs Power
    """

    fig = px.scatter(
        df,
        x="temperature_2_m_above_gnd",
        y="generated_power_kw",
        color="generated_power_kw",
        title="Temperature vs Generated Power",
        trendline="ols"
    )

    return fig


def humidity_vs_power(df):
    """
    Humidity vs Power
    """

    fig = px.scatter(
        df,
        x="relative_humidity_2_m_above_gnd",
        y="generated_power_kw",
        color="generated_power_kw",
        title="Humidity vs Generated Power"
    )

    return fig


def cloud_cover_vs_power(df):
    """
    Cloud Cover vs Power
    """

    fig = px.scatter(
        df,
        x="total_cloud_cover_sfc",
        y="generated_power_kw",
        color="generated_power_kw",
        title="Cloud Cover vs Generated Power"
    )

    return fig


def wind_speed_vs_power(df):
    """
    Wind Speed vs Power
    """

    fig = px.scatter(
        df,
        x="wind_speed_10_m_above_gnd",
        y="generated_power_kw",
        color="generated_power_kw",
        title="Wind Speed vs Generated Power"
    )

    return fig


def correlation_heatmap(df):
    """
    Correlation Heatmap
    """

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    return fig


def power_distribution(df):
    """
    Distribution of Generated Power
    """

    fig = px.histogram(
        df,
        x="generated_power_kw",
        nbins=30,
        title="Power Distribution"
    )

    return fig


def boxplot_power(df):
    """
    Boxplot for Power Generation
    """

    fig = px.box(
        df,
        y="generated_power_kw",
        title="Power Generation Boxplot"
    )

    return fig


def feature_importance_chart(feature_names,
                             importance_values):
    """
    Feature Importance Chart
    """

    importance_df = pd.DataFrame({

        "Feature": feature_names,
        "Importance": importance_values

    }).sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Important Features"
    )

    return fig


def actual_vs_predicted(y_test,
                        y_pred):
    """
    Model Performance Chart
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=y_test,
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=y_pred,
            mode="lines",
            name="Predicted"
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Power",
        xaxis_title="Samples",
        yaxis_title="Power (kW)"
    )

    return fig


def monthly_generation_chart(df):
    """
    Monthly Average Power
    """

    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])

    monthly = (
        df.groupby(
            df["date"].dt.month
        )["generated_power_kw"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        monthly,
        x="date",
        y="generated_power_kw",
        title="Monthly Average Power Generation"
    )

    return fig
