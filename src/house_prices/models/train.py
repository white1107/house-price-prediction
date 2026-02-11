"""Model training pipeline with MLflow tracking."""

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from house_prices.data.loader import load_raw_data, prepare_data
from house_prices.features.engineering import build_features
from house_prices.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_models(config: dict) -> dict:
    """Initialize models from config."""
    mc = config["model"]
    models = {
        "Ridge": Ridge(alpha=mc["ridge"]["alpha"]),
        "Lasso": Lasso(alpha=mc["lasso"]["alpha"]),
        "ElasticNet": ElasticNet(
            alpha=mc["elasticnet"]["alpha"],
            l1_ratio=mc["elasticnet"]["l1_ratio"],
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=mc["random_forest"]["n_estimators"],
            max_depth=mc["random_forest"]["max_depth"],
            min_samples_split=mc["random_forest"]["min_samples_split"],
            random_state=mc["random_state"],
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=mc["gradient_boosting"]["n_estimators"],
            learning_rate=mc["gradient_boosting"]["learning_rate"],
            max_depth=mc["gradient_boosting"]["max_depth"],
            min_samples_split=mc["gradient_boosting"]["min_samples_split"],
            random_state=mc["random_state"],
        ),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=mc["xgboost"]["n_estimators"],
            learning_rate=mc["xgboost"]["learning_rate"],
            max_depth=mc["xgboost"]["max_depth"],
            subsample=mc["xgboost"]["subsample"],
            colsample_bytree=mc["xgboost"]["colsample_bytree"],
            random_state=mc["random_state"],
            verbosity=0,
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=mc["lightgbm"]["n_estimators"],
            learning_rate=mc["lightgbm"]["learning_rate"],
            max_depth=mc["lightgbm"]["max_depth"],
            num_leaves=mc["lightgbm"]["num_leaves"],
            subsample=mc["lightgbm"]["subsample"],
            random_state=mc["random_state"],
            verbosity=-1,
        )
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    return models


def train_and_evaluate(config: dict) -> tuple[str, object, StandardScaler]:
    """Train all models, evaluate with CV, and return the best one."""
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]

    # Load and prepare data
    train_df, test_df = load_raw_data(str(raw_dir))
    combined, target, test_ids = prepare_data(
        train_df, test_df, log_transform=config["preprocessing"]["target"]["log_transform"]
    )

    # Feature engineering
    combined = build_features(combined)

    # Split back
    n_train = len(target)
    X_train = combined.iloc[:n_train]
    X_test = combined.iloc[n_train:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    # MLflow setup
    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Train and evaluate all models
    models = get_models(config)
    results = {}
    best_score = float("inf")
    best_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            scores = cross_val_score(
                model,
                X_train_scaled,
                target,
                cv=config["model"]["cv_folds"],
                scoring=config["model"]["scoring"],
            )
            rmse_scores = -scores
            mean_rmse = rmse_scores.mean()
            std_rmse = rmse_scores.std()

            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metric("cv_rmse_mean", mean_rmse)
            mlflow.log_metric("cv_rmse_std", std_rmse)
            for i, s in enumerate(rmse_scores):
                mlflow.log_metric(f"cv_rmse_fold_{i}", s)

            results[name] = {"mean": mean_rmse, "std": std_rmse, "scores": rmse_scores}
            logger.info(f"{name}: RMSE = {mean_rmse:.5f} (+/- {std_rmse:.5f})")

            if mean_rmse < best_score:
                best_score = mean_rmse
                best_name = name

    logger.info(f"\nBest model: {best_name} (RMSE: {best_score:.5f})")

    # Retrain best model on full training data
    best_model = models[best_name]
    best_model.fit(X_train_scaled, target)

    # Save model and scaler
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(X_train.columns.tolist(), models_dir / "feature_names.joblib")
    joblib.dump(test_ids, models_dir / "test_ids.joblib")
    logger.info(f"Saved model artifacts to {models_dir}")

    # Log best model to MLflow (metrics only, skip heavy model serialization)
    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.log_params(best_model.get_params())
        mlflow.log_metric("cv_rmse_mean", best_score)

    return best_name, best_model, scaler


if __name__ == "__main__":
    config = load_config()
    best_name, best_model, scaler = train_and_evaluate(config)
    print(f"\nTraining complete. Best model: {best_name}")
