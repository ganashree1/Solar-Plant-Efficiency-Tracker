import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load CSV dataset
    """
    df = pd.read_csv(file_path)

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    return df


def clean_data(df):
    """
    Data Cleaning
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def feature_engineering(df):
    """
    Create additional features
    """

    # Solar Efficiency Index
    if (
        "generated_power_kw" in df.columns
        and "shortwave_radiation_backwards_sfc" in df.columns
    ):
        df["efficiency_index"] = (
            df["generated_power_kw"]
            / (df["shortwave_radiation_backwards_sfc"] + 1)
        )

    # Temperature Category
    if "temperature_2_m_above_gnd" in df.columns:

        df["temp_category"] = pd.cut(
            df["temperature_2_m_above_gnd"],
            bins=[-50, 15, 25, 35, 60],
            labels=[
                "Low",
                "Moderate",
                "High",
                "Very High"
            ]
        )

    return df


def remove_outliers(df):
    """
    Remove outliers using IQR method
    """

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[
            (df[col] >= lower)
            & (df[col] <= upper)
        ]

    return df


def preprocess_data(file_path):
    """
    Complete preprocessing pipeline
    """

    df = load_data(file_path)

    df = clean_data(df)

    df = feature_engineering(df)

    df = remove_outliers(df)

    return df


if __name__ == "__main__":

    data = preprocess_data("data/spg.csv")

    print("Dataset Shape:", data.shape)

    print(data.head())
