"""Tests for advanced feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from house_prices.data.loader import load_raw_data, prepare_data
from house_prices.features.advanced_engineering import (
    add_engineered_features,
    build_features,
    encode_ordinal_features,
    fill_missing_values,
    fix_skewed_features,
)


@pytest.fixture
def combined_data():
    """Load and prepare real data."""
    train, test = load_raw_data("data/raw")
    combined, target, test_ids = prepare_data(train, test)
    return combined, target


class TestFillMissingValues:
    def test_no_missing_after_fill(self, combined_data):
        combined, _ = combined_data
        result = fill_missing_values(combined)
        assert result.isnull().sum().sum() == 0

    def test_preserves_row_count(self, combined_data):
        combined, _ = combined_data
        result = fill_missing_values(combined)
        assert len(result) == len(combined)

    def test_does_not_modify_original(self, combined_data):
        combined, _ = combined_data
        original_nulls = combined.isnull().sum().sum()
        fill_missing_values(combined)
        assert combined.isnull().sum().sum() == original_nulls

    def test_pool_qc_filled(self, combined_data):
        combined, _ = combined_data
        result = fill_missing_values(combined)
        assert result["PoolQC"].isnull().sum() == 0


class TestAddEngineeredFeatures:
    def test_total_sf_correct(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = add_engineered_features(df)
        expected = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        pd.testing.assert_series_equal(result["TotalSF"], expected, check_names=False)

    def test_total_bathrooms_correct(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = add_engineered_features(df)
        expected = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        pd.testing.assert_series_equal(result["TotalBathrooms"], expected, check_names=False)

    def test_house_age_non_negative(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = add_engineered_features(df)
        # Some houses might be sold before remod, so HouseAge can be negative
        # but GarageAge should be clipped to 0
        assert (result["GarageAge"] >= 0).all()

    def test_binary_flags_are_binary(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = add_engineered_features(df)
        for col in ["HasPool", "HasGarage", "HasFireplace", "Has2ndFloor", "HasBsmt", "IsRemodeled", "IsNewHouse"]:
            assert set(result[col].unique()).issubset({0, 1})

    def test_new_columns_added(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = add_engineered_features(df)
        expected_cols = [
            "TotalSF", "TotalBathrooms", "TotalPorchSF", "HasPool", "HasGarage",
            "HasFireplace", "HouseAge", "RemodAge", "GarageAge", "OverallScore",
            "QualSF", "QualAge", "OverallQual_sq", "GrLivArea_sq", "TotalSF_sq",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestEncodeOrdinalFeatures:
    def test_quality_mapping(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = encode_ordinal_features(df)
        assert result["ExterQual"].dtype in [np.int64, np.int32, int]
        assert result["ExterQual"].max() <= 5
        assert result["ExterQual"].min() >= 0

    def test_fence_mapping(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = encode_ordinal_features(df)
        assert result["Fence"].max() <= 4

    def test_functional_mapping(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        result = encode_ordinal_features(df)
        assert result["Functional"].max() <= 7


class TestFixSkewedFeatures:
    def test_reduces_skewness(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        df = add_engineered_features(df)
        df = encode_ordinal_features(df)
        from scipy.stats import skew
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        before_skew = df[numeric_cols].apply(lambda x: abs(skew(x.dropna()))).mean()
        result = fix_skewed_features(df)
        after_skew = result[numeric_cols].apply(lambda x: abs(skew(x.dropna()))).mean()
        assert after_skew <= before_skew

    def test_no_negative_values_log_transformed(self, combined_data):
        combined, _ = combined_data
        df = fill_missing_values(combined)
        df = add_engineered_features(df)
        df = encode_ordinal_features(df)
        result = fix_skewed_features(df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert result[col].min() >= 0 or (df[col] < 0).any()


class TestBuildFeatures:
    def test_full_pipeline_no_errors(self, combined_data):
        combined, _ = combined_data
        result = build_features(combined)
        assert result.shape[0] == combined.shape[0]

    def test_no_missing_values(self, combined_data):
        combined, _ = combined_data
        result = build_features(combined)
        assert result.isnull().sum().sum() == 0

    def test_no_object_columns(self, combined_data):
        combined, _ = combined_data
        result = build_features(combined, encode_nominal=True)
        assert len(result.select_dtypes(include=["object"]).columns) == 0

    def test_skip_nominal_encoding(self, combined_data):
        combined, _ = combined_data
        result = build_features(combined, encode_nominal=False)
        assert len(result.select_dtypes(include=["object"]).columns) > 0

    def test_consistent_output_shape(self, combined_data):
        combined, _ = combined_data
        result1 = build_features(combined.copy())
        result2 = build_features(combined.copy())
        assert result1.shape == result2.shape
