import pandas as pd
import numpy as np


def generate_insights(df):
    """
    Generate automatic insights from dataset
    """

    insights = []

    # ------------------------------------------------
    # Correlation Analysis
    # ------------------------------------------------

    corr = df.corr(numeric_only=True)

    if "generated_power_kw" in corr.columns:

        power_corr = corr["generated_power_kw"].drop(
            "generated_power_kw"
        )

        strongest_factor = power_corr.abs().idxmax()

        strongest_value = power_corr[
            strongest_factor
        ]

        insights.append(
            f"Strongest factor affecting power generation is "
            f"'{strongest_factor}' "
            f"(Correlation = {strongest_value:.2f})."
        )

    # ------------------------------------------------
    # Radiation Analysis
    # ------------------------------------------------

    if (
        "shortwave_radiation_backwards_sfc"
        in df.columns
    ):

        avg_radiation = df[
            "shortwave_radiation_backwards_sfc"
        ].mean()

        insights.append(
            f"Average solar radiation observed is "
            f"{avg_radiation:.2f} W/m²."
        )

    # ------------------------------------------------
    # Temperature Analysis
    # ------------------------------------------------

    if (
        "temperature_2_m_above_gnd"
        in df.columns
    ):

        avg_temp = df[
            "temperature_2_m_above_gnd"
        ].mean()

        insights.append(
            f"Average operating temperature is "
            f"{avg_temp:.2f} °C."
        )

    # ------------------------------------------------
    # Cloud Cover Analysis
    # ------------------------------------------------

    if "total_cloud_cover_sfc" in df.columns:

        avg_cloud = df[
            "total_cloud_cover_sfc"
        ].mean()

        if avg_cloud > 60:

            insights.append(
                "High cloud cover may be reducing "
                "solar power generation."
            )

        else:

            insights.append(
                "Cloud cover impact appears moderate."
            )

    # ------------------------------------------------
    # Humidity Analysis
    # ------------------------------------------------

    if (
        "relative_humidity_2_m_above_gnd"
        in df.columns
    ):

        avg_humidity = df[
            "relative_humidity_2_m_above_gnd"
        ].mean()

        insights.append(
            f"Average humidity level is "
            f"{avg_humidity:.2f}%."
        )

    # ------------------------------------------------
    # Power Generation Analysis
    # ------------------------------------------------

    avg_power = df["generated_power_kw"].mean()

    max_power = df["generated_power_kw"].max()

    insights.append(
        f"Average generated power is "
        f"{avg_power:.2f} kW."
    )

    insights.append(
        f"Peak generated power reached "
        f"{max_power:.2f} kW."
    )

    # ------------------------------------------------
    # Efficiency Score
    # ------------------------------------------------

    if (
        "generated_power_kw" in df.columns
        and
        "shortwave_radiation_backwards_sfc"
        in df.columns
    ):

        efficiency = (
            df["generated_power_kw"]
            /
            (
                df[
                    "shortwave_radiation_backwards_sfc"
                ] + 1
            )
        ).mean()

        efficiency_score = efficiency * 100

        insights.append(
            f"Estimated solar efficiency score "
            f"is {efficiency_score:.2f}%."
        )

    return insights


def executive_summary(df):
    """
    Generate executive summary
    """

    total_power = df[
        "generated_power_kw"
    ].sum()

    avg_power = df[
        "generated_power_kw"
    ].mean()

    summary = f"""
    Total Power Generated:
    {total_power:.2f} kW

    Average Power Output:
    {avg_power:.2f} kW

    The solar plant is operating under
    monitored environmental conditions.
    Solar radiation remains the key
    factor influencing power output.
    """

    return summary


def recommendation_engine(df):
    """
    Generate recommendations
    """

    recommendations = []

    if (
        "total_cloud_cover_sfc"
        in df.columns
    ):

        if (
            df["total_cloud_cover_sfc"].mean()
            > 60
        ):

            recommendations.append(
                "Monitor cloudy periods and plan "
                "energy storage accordingly."
            )

    if (
        "shortwave_radiation_backwards_sfc"
        in df.columns
    ):

        recommendations.append(
            "Maximize production during peak "
            "solar radiation hours."
        )

    recommendations.append(
        "Perform regular panel cleaning to "
        "improve efficiency."
    )

    recommendations.append(
        "Monitor weather forecasts to "
        "predict energy generation."
    )

    recommendations.append(
        "Use machine learning forecasts for "
        "power planning and load balancing."
    )

    return recommendations


if __name__ == "__main__":

    df = pd.read_csv("data/spg.csv")

    print("\nAI INSIGHTS")
    print("=" * 50)

    for insight in generate_insights(df):
        print("✓", insight)

    print("\nEXECUTIVE SUMMARY")
    print("=" * 50)

    print(executive_summary(df))

    print("\nRECOMMENDATIONS")
    print("=" * 50)

    for rec in recommendation_engine(df):
        print("✓", rec)
