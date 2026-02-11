"""Feature engineering pipeline."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with domain-specific logic."""
    df = df.copy()

    # Step 1: NA means "no feature" for these columns
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType",
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath",
        "MasVnrArea",
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Step 2: LotFrontage - fill with neighborhood median
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    # Step 3: Remaining
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    logger.info(f"Missing values after fill: {df.isnull().sum().sum()}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones."""
    df = df.copy()

    # Area aggregations
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"]
        + df["3SsnPorch"] + df["ScreenPorch"]
    )

    # Binary flags
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # Age features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    logger.info(f"Added 8 engineered features. New shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features."""
    df = df.copy()

    # Ordinal encoding
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    quality_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC",
    ]
    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].map(quality_map).fillna(0).astype(int)

    bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map(bsmt_exposure_map).fillna(0).astype(int)

    bsmt_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col] = df[col].map(bsmt_fin_map).fillna(0).astype(int)

    garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
    if "GarageFinish" in df.columns:
        df["GarageFinish"] = df["GarageFinish"].map(garage_finish_map).fillna(0).astype(int)

    fence_map = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
    if "Fence" in df.columns:
        df["Fence"] = df["Fence"].map(fence_map).fillna(0).astype(int)

    # One-hot encoding for remaining categoricals
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        logger.info(f"One-hot encoded {len(categorical_cols)} columns. Shape: {df.shape}")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = fill_missing_values(df)
    df = add_engineered_features(df)
    df = encode_features(df)
    return df
