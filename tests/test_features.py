"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from house_prices.features.engineering import (
    add_engineered_features,
    encode_features,
    fill_missing_values,
)


@pytest.fixture
def sample_train_data():
    """Create minimal sample data mimicking the House Prices dataset."""
    return pd.DataFrame({
        "LotFrontage": [65.0, np.nan, 70.0, np.nan],
        "Neighborhood": ["NAmes", "NAmes", "CollgCr", "CollgCr"],
        "OverallQual": [7, 6, 8, 5],
        "TotalBsmtSF": [856, 1262, 920, 0],
        "1stFlrSF": [856, 1262, 920, 961],
        "2ndFlrSF": [854, 0, 866, 0],
        "FullBath": [2, 2, 2, 1],
        "HalfBath": [1, 0, 1, 0],
        "BsmtFullBath": [1, 0, 1, 0],
        "BsmtHalfBath": [0, 1, 0, 0],
        "GarageArea": [548, 460, 608, 0],
        "PoolArea": [0, 0, 0, 0],
        "Fireplaces": [0, 1, 1, 0],
        "OpenPorchSF": [61, 0, 42, 35],
        "EnclosedPorch": [0, 0, 0, 0],
        "3SsnPorch": [0, 0, 0, 0],
        "ScreenPorch": [0, 0, 0, 0],
        "YrSold": [2008, 2007, 2008, 2009],
        "YearBuilt": [2003, 1976, 2001, 1970],
        "YearRemodAdd": [2003, 1976, 2002, 1970],
        "PoolQC": [np.nan, np.nan, np.nan, np.nan],
        "MiscFeature": [np.nan, np.nan, np.nan, np.nan],
        "Alley": [np.nan, np.nan, np.nan, np.nan],
        "Fence": [np.nan, np.nan, "GdPrv", np.nan],
        "FireplaceQu": [np.nan, "TA", "Gd", np.nan],
        "GarageType": ["Attchd", "Attchd", "Attchd", np.nan],
        "GarageFinish": ["RFn", "RFn", "Fin", np.nan],
        "GarageQual": ["TA", "TA", "TA", np.nan],
        "GarageCond": ["TA", "TA", "TA", np.nan],
        "BsmtQual": ["Gd", "Gd", "Ex", np.nan],
        "BsmtCond": ["TA", "TA", "TA", np.nan],
        "BsmtExposure": ["No", "Gd", "Mn", np.nan],
        "BsmtFinType1": ["GLQ", "ALQ", "GLQ", np.nan],
        "BsmtFinType2": ["Unf", "Unf", "Unf", np.nan],
        "MasVnrType": ["BrkFace", np.nan, "None", np.nan],
        "MasVnrArea": [196.0, np.nan, 0.0, np.nan],
        "ExterQual": ["Gd", "TA", "Ex", "TA"],
        "ExterCond": ["TA", "TA", "TA", "TA"],
        "HeatingQC": ["Ex", "Ex", "Ex", "Gd"],
        "KitchenQual": ["Gd", "TA", "Ex", "TA"],
        "GarageYrBlt": [2003.0, 1976.0, 2001.0, np.nan],
        "BsmtFinSF1": [706, 978, 486, np.nan],
        "BsmtFinSF2": [0, 0, 0, np.nan],
        "BsmtUnfSF": [150, 284, 434, np.nan],
        "BsmtHalfBath": [0, 1, 0, 0],
        "GarageCars": [2, 2, 2, np.nan],
        "MSZoning": ["RL", "RL", "RL", "RM"],
    })


class TestFillMissingValues:
    def test_no_missing_after_fill(self, sample_train_data):
        result = fill_missing_values(sample_train_data)
        assert result.isnull().sum().sum() == 0

    def test_pool_qc_filled_with_none(self, sample_train_data):
        result = fill_missing_values(sample_train_data)
        assert (result["PoolQC"] == "None").all()

    def test_garage_yr_blt_filled_with_zero(self, sample_train_data):
        result = fill_missing_values(sample_train_data)
        assert result["GarageYrBlt"].iloc[3] == 0

    def test_lot_frontage_filled_by_neighborhood(self, sample_train_data):
        result = fill_missing_values(sample_train_data)
        # NAmes neighborhood: median of [65.0] = 65.0
        assert result["LotFrontage"].iloc[1] == 65.0

    def test_does_not_modify_original(self, sample_train_data):
        original_nulls = sample_train_data.isnull().sum().sum()
        fill_missing_values(sample_train_data)
        assert sample_train_data.isnull().sum().sum() == original_nulls


class TestAddEngineeredFeatures:
    def test_total_sf_calculated(self, sample_train_data):
        df = fill_missing_values(sample_train_data)
        result = add_engineered_features(df)
        expected = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        pd.testing.assert_series_equal(result["TotalSF"], expected, check_names=False)

    def test_house_age_calculated(self, sample_train_data):
        df = fill_missing_values(sample_train_data)
        result = add_engineered_features(df)
        assert result["HouseAge"].iloc[0] == 2008 - 2003

    def test_has_garage_flag(self, sample_train_data):
        df = fill_missing_values(sample_train_data)
        result = add_engineered_features(df)
        assert result["HasGarage"].iloc[0] == 1
        assert result["HasGarage"].iloc[3] == 0

    def test_new_columns_added(self, sample_train_data):
        df = fill_missing_values(sample_train_data)
        result = add_engineered_features(df)
        expected_cols = [
            "TotalSF", "TotalBathrooms", "TotalPorchSF",
            "HasPool", "HasGarage", "HasFireplace",
            "HouseAge", "RemodAge",
        ]
        for col in expected_cols:
            assert col in result.columns


class TestEncodeFeatures:
    def test_ordinal_encoding(self, sample_train_data):
        df = fill_missing_values(sample_train_data)
        result = encode_features(df)
        # ExterQual: Gd -> 4, TA -> 3, Ex -> 5
        assert result["ExterQual"].iloc[0] == 4
        assert result["ExterQual"].iloc[2] == 5

    def test_no_object_columns_after_encoding(self, sample_train_data):
        df = fill_missing_values(sample_train_data)
        result = encode_features(df)
        assert len(result.select_dtypes(include=["object"]).columns) == 0

    def test_fence_ordinal_encoding(self, sample_train_data):
        df = fill_missing_values(sample_train_data)
        result = encode_features(df)
        # Fence: GdPrv -> 4, None -> 0
        assert result["Fence"].iloc[2] == 4
        assert result["Fence"].iloc[0] == 0
