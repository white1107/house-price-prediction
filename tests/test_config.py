"""Tests for configuration loading."""

import pytest

from house_prices.utils.config import load_config


class TestLoadConfig:
    def test_loads_successfully(self):
        config = load_config()
        assert isinstance(config, dict)

    def test_has_data_section(self):
        config = load_config()
        assert "data" in config
        assert "raw_dir" in config["data"]
        assert "train_file" in config["data"]

    def test_has_model_section(self):
        config = load_config()
        assert "model" in config
        assert "cv_folds" in config["model"]
        assert "random_state" in config["model"]

    def test_has_mlflow_section(self):
        config = load_config()
        assert "mlflow" in config
        assert "experiment_name" in config["mlflow"]

    def test_has_output_section(self):
        config = load_config()
        assert "output" in config
        assert "submissions_dir" in config["output"]
        assert "models_dir" in config["output"]

    def test_model_hyperparams_exist(self):
        config = load_config()
        for model_name in ["ridge", "lasso", "elasticnet", "xgboost", "lightgbm"]:
            assert model_name in config["model"], f"Missing config for {model_name}"

    def test_cv_folds_reasonable(self):
        config = load_config()
        assert 2 <= config["model"]["cv_folds"] <= 10

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
