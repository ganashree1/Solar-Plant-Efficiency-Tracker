import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_kpis(df):
    """
    Calculate dashboard KPI values
    """

    kpis = {

        "Total Power":
            round(df["generated_power_kw"].sum(), 2),

        "Average Power":
            round(df["generated_power_kw"].mean(), 2),

        "Maximum Power":
            round(df["generated_power_kw"].max(), 2),

        "Minimum Power":
            round(df["generated_power_kw"].min(), 2),

        "Average Temperature":
            round(
                df["temperature_2_m_above_gnd"].mean(),
                2
            ),

        "Average Radiation":
            round(
                df["shortwave_radiation_backwards_sfc"].mean(),
                2
            )
    }

    return kpis


def dataset_summary(df):
    """
    Dataset overview
    """

    summary = {

        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values":
            df.isnull().sum().sum(),
        "Duplicates":
            df.duplicated().sum()
    }

    return summary


def descriptive_statistics(df):
    """
    Descriptive statistics
    """

    return df.describe()


def correlation_matrix(df):
    """
    Correlation matrix
    """

    return df.corr(numeric_only=True)


def power_correlations(df):
    """
    Features correlation with power output
    """

    corr = (
        df.corr(numeric_only=True)
        ["generated_power_kw"]
        .sort_values(ascending=False)
    )

    return corr


def top_positive_factors(df, n=5):
    """
    Top positive factors affecting power
    """

    corr = power_correlations(df)

    return corr[1:n+1]


def top_negative_factors(df, n=5):
    """
    Top negative factors affecting power
    """

    corr = power_correlations(df)

    return corr.sort_values().head(n)


def calculate_efficiency_score(df):
    """
    Solar efficiency score
    """

    if (
        "generated_power_kw" in df.columns
        and "shortwave_radiation_backwards_sfc"
        in df.columns
    ):

        efficiency = (
            df["generated_power_kw"]
            /
            (df["shortwave_radiation_backwards_sfc"] + 1)
        ).mean()

        return round(efficiency * 100, 2)

    return 0


def evaluate_model(y_true, y_pred):
    """
    Model evaluation metrics
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    metrics = {

        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 3)
    }

    return metrics


def monthly_power_analysis(df):
    """
    Monthly power generation analysis
    """

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        )

        monthly = (
            df.groupby(
                df["date"].dt.month
            )["generated_power_kw"]
            .mean()
            .reset_index()
        )

        return monthly

    return pd.DataFrame()


def weather_impact_analysis(df):
    """
    Weather impact on solar generation
    """

    weather_columns = [

        "temperature_2_m_above_gnd",
        "relative_humidity_2_m_above_gnd",
        "wind_speed_10_m_above_gnd",
        "total_cloud_cover_sfc",
        "shortwave_radiation_backwards_sfc"
    ]

    available_cols = [

        col for col in weather_columns
        if col in df.columns
    ]

    impact = (
        df[available_cols +
           ["generated_power_kw"]]
        .corr()
        ["generated_power_kw"]
        .sort_values(ascending=False)
    )

    return impact


if __name__ == "__main__":

    df = pd.read_csv("data/spg.csv")

    print("\nDataset Summary")
    print(dataset_summary(df))

    print("\nKPIs")
    print(calculate_kpis(df))

    print("\nCorrelation Analysis")
    print(power_correlations(df))

    print("\nEfficiency Score")
    print(calculate_efficiency_score(df))
