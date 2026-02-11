"""Advanced feature engineering for House Prices competition."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import skew

logger = logging.getLogger(__name__)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with domain-specific logic."""
    df = df.copy()

    # NA means "no feature"
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

    # LotFrontage - neighborhood median
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    # Remaining
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "None")
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)

    logger.info(f"Missing values after fill: {df.isnull().sum().sum()}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive engineered features."""
    df = df.copy()

    # === Area features ===
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"]
        + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["LivingAreaRatio"] = df["GrLivArea"] / (df["LotArea"] + 1)
    df["GarageAreaPerCar"] = df["GarageArea"] / (df["GarageCars"] + 1)
    df["BsmtFinRatio"] = df["BsmtFinSF1"] / (df["TotalBsmtSF"] + 1)

    # === Binary flags ===
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasMasVnr"] = (df["MasVnrArea"] > 0).astype(int)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNewHouse"] = (df["YrSold"] == df["YearBuilt"]).astype(int)

    # === Age features ===
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["GarageAge"] = df["GarageAge"].clip(lower=0)

    # === Quality interaction features ===
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    df["QualSF"] = df["OverallQual"] * df["TotalSF"]
    df["QualAge"] = df["OverallQual"] * df["HouseAge"]

    # === Neighborhood aggregations (relative features) ===
    neigh_median = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    df["GrLivArea_NeighRatio"] = df["GrLivArea"] / (neigh_median + 1)

    neigh_lot_median = df.groupby("Neighborhood")["LotArea"].transform("median")
    df["LotArea_NeighRatio"] = df["LotArea"] / (neigh_lot_median + 1)

    # === Polynomial features for top correlated ===
    df["OverallQual_sq"] = df["OverallQual"] ** 2
    df["GrLivArea_sq"] = df["GrLivArea"] ** 2
    df["TotalSF_sq"] = df["TotalSF"] ** 2

    logger.info(f"Engineered features added. Shape: {df.shape}")
    return df


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal categorical features to numeric."""
    df = df.copy()

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

    functional_map = {
        "Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
        "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0,
    }
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].map(functional_map).fillna(7).astype(int)

    lot_shape_map = {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0}
    if "LotShape" in df.columns:
        df["LotShape"] = df["LotShape"].map(lot_shape_map).fillna(0).astype(int)

    land_slope_map = {"Gtl": 2, "Mod": 1, "Sev": 0}
    if "LandSlope" in df.columns:
        df["LandSlope"] = df["LandSlope"].map(land_slope_map).fillna(0).astype(int)

    paved_map = {"Y": 2, "P": 1, "N": 0}
    if "PavedDrive" in df.columns:
        df["PavedDrive"] = df["PavedDrive"].map(paved_map).fillna(0).astype(int)

    street_map = {"Pave": 1, "Grvl": 0}
    if "Street" in df.columns:
        df["Street"] = df["Street"].map(street_map).fillna(0).astype(int)

    central_air_map = {"Y": 1, "N": 0}
    if "CentralAir" in df.columns:
        df["CentralAir"] = df["CentralAir"].map(central_air_map).fillna(0).astype(int)

    return df


def fix_skewed_features(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Apply log1p to highly skewed numerical features."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed = df[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_features = skewed[abs(skewed) > threshold].index.tolist()

    for col in skewed_features:
        if (df[col] >= 0).all():
            df[col] = np.log1p(df[col])

    logger.info(f"Log-transformed {len(skewed_features)} skewed features")
    return df


def encode_nominal_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining categorical features."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        logger.info(f"One-hot encoded {len(categorical_cols)} nominal features. Shape: {df.shape}")
    return df


def build_features(
    df: pd.DataFrame,
    encode_nominal: bool = True,
    fix_skew: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Args:
        df: Combined train+test DataFrame
        encode_nominal: Whether to one-hot encode nominal features
            (set False for CatBoost which handles categoricals natively)
        fix_skew: Whether to apply log transform to skewed features
    """
    df = fill_missing_values(df)
    df = add_engineered_features(df)
    df = encode_ordinal_features(df)
    if fix_skew:
        df = fix_skewed_features(df)
    if encode_nominal:
        df = encode_nominal_features(df)
    return df
