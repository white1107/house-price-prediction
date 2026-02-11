"""Tests for data loading and preparation."""

import numpy as np
import pandas as pd
import pytest

from house_prices.data.loader import load_raw_data, prepare_data


@pytest.fixture
def raw_data():
    """Load real raw data."""
    return load_raw_data("data/raw")


class TestLoadRawData:
    def test_returns_two_dataframes(self, raw_data):
        train, test = raw_data
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_train_has_target(self, raw_data):
        train, _ = raw_data
        assert "SalePrice" in train.columns

    def test_test_has_no_target(self, raw_data):
        _, test = raw_data
        assert "SalePrice" not in test.columns

    def test_train_shape(self, raw_data):
        train, _ = raw_data
        assert train.shape[0] == 1460
        assert train.shape[1] == 81

    def test_test_shape(self, raw_data):
        _, test = raw_data
        assert test.shape[0] == 1459
        assert test.shape[1] == 80

    def test_both_have_id(self, raw_data):
        train, test = raw_data
        assert "Id" in train.columns
        assert "Id" in test.columns


class TestPrepareData:
    def test_outlier_removal(self, raw_data):
        train, test = raw_data
        combined, target, _ = prepare_data(train, test)
        n_outliers = (train["GrLivArea"] > 4000).sum()
        assert len(target) == len(train) - n_outliers

    def test_log_transform_applied(self, raw_data):
        train, test = raw_data
        _, target, _ = prepare_data(train, test, log_transform=True)
        # log-transformed values should be much smaller than raw prices
        assert target.max() < 20  # log1p(800000) ~ 13.6

    def test_no_log_transform(self, raw_data):
        train, test = raw_data
        _, target, _ = prepare_data(train, test, log_transform=False)
        assert target.max() > 100000

    def test_combined_shape(self, raw_data):
        train, test = raw_data
        combined, target, _ = prepare_data(train, test)
        n_outliers = (train["GrLivArea"] > 4000).sum()
        expected_rows = len(train) - n_outliers + len(test)
        assert combined.shape[0] == expected_rows

    def test_no_id_or_target_in_combined(self, raw_data):
        train, test = raw_data
        combined, _, _ = prepare_data(train, test)
        assert "Id" not in combined.columns
        assert "SalePrice" not in combined.columns

    def test_test_ids_preserved(self, raw_data):
        train, test = raw_data
        _, _, test_ids = prepare_data(train, test)
        assert len(test_ids) == len(test)
        assert test_ids.iloc[0] == test["Id"].iloc[0]
