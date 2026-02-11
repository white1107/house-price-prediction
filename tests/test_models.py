"""Tests for model training and prediction pipeline."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.linear_model import Ridge

from house_prices.data.loader import load_raw_data, prepare_data
from house_prices.features.advanced_engineering import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline_data():
    """Prepare data through full pipeline (cached per module)."""
    train, test = load_raw_data("data/raw")
    combined, target, test_ids = prepare_data(train, test)
    combined = build_features(combined)
    n_train = len(target)
    X_train = combined.iloc[:n_train]
    X_test = combined.iloc[n_train:]
    return X_train, X_test, target, test_ids


class TestPipelineData:
    def test_no_missing_values_in_train(self, pipeline_data):
        X_train, _, _, _ = pipeline_data
        assert X_train.isnull().sum().sum() == 0

    def test_no_missing_values_in_test(self, pipeline_data):
        _, X_test, _, _ = pipeline_data
        assert X_test.isnull().sum().sum() == 0

    def test_train_test_same_columns(self, pipeline_data):
        X_train, X_test, _, _ = pipeline_data
        assert list(X_train.columns) == list(X_test.columns)

    def test_target_length_matches_train(self, pipeline_data):
        X_train, _, target, _ = pipeline_data
        assert len(target) == len(X_train)

    def test_no_infinite_values(self, pipeline_data):
        X_train, X_test, _, _ = pipeline_data
        assert not np.isinf(X_train.values).any()
        assert not np.isinf(X_test.values).any()


class TestModelTraining:
    def test_ridge_can_fit(self, pipeline_data):
        X_train, _, target, _ = pipeline_data
        model = Ridge(alpha=10.0)
        model.fit(X_train, target)
        assert hasattr(model, "coef_")

    def test_ridge_predictions_reasonable(self, pipeline_data):
        X_train, X_test, target, _ = pipeline_data
        model = Ridge(alpha=10.0)
        model.fit(X_train, target)
        preds = model.predict(X_test)
        # Log-scale predictions should be in range [8, 15] roughly
        assert preds.min() > 5
        assert preds.max() < 20

    def test_ridge_train_score_reasonable(self, pipeline_data):
        X_train, _, target, _ = pipeline_data
        model = Ridge(alpha=10.0)
        model.fit(X_train, target)
        score = model.score(X_train, target)
        assert score > 0.7  # R^2 should be reasonable


class TestSavedModel:
    def test_model_file_exists(self):
        assert (PROJECT_ROOT / "models" / "best_model.joblib").exists()

    def test_scaler_file_exists(self):
        assert (PROJECT_ROOT / "models" / "scaler.joblib").exists()

    def test_feature_names_file_exists(self):
        assert (PROJECT_ROOT / "models" / "feature_names.joblib").exists()

    def test_saved_model_can_predict(self):
        import joblib
        model = joblib.load(PROJECT_ROOT / "models" / "best_model.joblib")
        scaler = joblib.load(PROJECT_ROOT / "models" / "scaler.joblib")
        feature_names = joblib.load(PROJECT_ROOT / "models" / "feature_names.joblib")

        # Create dummy input
        dummy = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        dummy_scaled = scaler.transform(dummy)
        pred = model.predict(dummy_scaled)
        assert len(pred) == 1
        assert np.isfinite(pred[0])
