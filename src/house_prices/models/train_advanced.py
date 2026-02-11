"""Advanced training pipeline: All models + Optuna + MLflow + Ensemble."""

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from house_prices.data.loader import load_raw_data, prepare_data
from house_prices.features.advanced_engineering import build_features
from house_prices.models.optuna_tuning import run_optuna
from house_prices.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_pipeline_data(config):
    """Load data and run feature engineering."""
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    train_df, test_df = load_raw_data(str(raw_dir))
    combined, target, test_ids = prepare_data(
        train_df, test_df, log_transform=config["preprocessing"]["target"]["log_transform"]
    )

    # Full FE with one-hot encoding (for tree/linear models)
    combined_encoded = build_features(combined.copy(), encode_nominal=True, fix_skew=True)

    n_train = len(target)
    X_train = combined_encoded.iloc[:n_train]
    X_test = combined_encoded.iloc[n_train:]

    # Scale for linear/NN models
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "target": target,
        "test_ids": test_ids,
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
    }


def train_all_models(config, data, n_trials=50):
    """Train all models with Optuna tuning."""
    X_train = data["X_train"]
    X_train_scaled = data["X_train_scaled"]
    target = data["target"]
    cv = config["model"]["cv_folds"]

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    results = {}

    # ========================================
    # 1. Linear Models (use scaled data)
    # ========================================
    logger.info("=" * 60)
    logger.info("Optimizing Linear Models...")
    logger.info("=" * 60)

    for name in ["Ridge", "Lasso", "ElasticNet"]:
        logger.info(f"\n--- {name} ---")
        opt = run_optuna(name, X_train_scaled, target, n_trials=n_trials, cv=cv)
        with mlflow.start_run(run_name=f"{name}_optuna"):
            mlflow.log_params(opt["best_params"])
            mlflow.log_metric("cv_rmse", opt["best_score"])
        results[name] = {
            "score": opt["best_score"],
            "params": opt["best_params"],
            "uses_scaled": True,
        }

    # ========================================
    # 2. Tree-Based Models (use unscaled data)
    # ========================================
    logger.info("=" * 60)
    logger.info("Optimizing Tree-Based Models...")
    logger.info("=" * 60)

    for name in ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting"]:
        logger.info(f"\n--- {name} ---")
        opt = run_optuna(name, X_train, target, n_trials=n_trials, cv=cv)
        with mlflow.start_run(run_name=f"{name}_optuna"):
            mlflow.log_params(opt["best_params"])
            mlflow.log_metric("cv_rmse", opt["best_score"])
        results[name] = {
            "score": opt["best_score"],
            "params": opt["best_params"],
            "uses_scaled": False,
        }

    # ========================================
    # 3. Deep Learning Models (use scaled data)
    # ========================================
    logger.info("=" * 60)
    logger.info("Training Deep Learning Models...")
    logger.info("=" * 60)

    from house_prices.models.deep_models import (
        FTTransformerRegressor,
        RealMLPRegressor,
        optimize_ft_transformer,
        optimize_realmlp,
    )

    # RealMLP with Optuna
    logger.info("\n--- RealMLP ---")
    study_mlp = __import__("optuna").create_study(direction="minimize", study_name="RealMLP")
    study_mlp.optimize(
        lambda trial: optimize_realmlp(trial, X_train_scaled.values, target.values, cv=cv),
        n_trials=max(10, n_trials // 3),
    )
    with mlflow.start_run(run_name="RealMLP_optuna"):
        mlflow.log_params(study_mlp.best_params)
        mlflow.log_metric("cv_rmse", study_mlp.best_value)
    results["RealMLP"] = {
        "score": study_mlp.best_value,
        "params": study_mlp.best_params,
        "uses_scaled": True,
    }
    logger.info(f"RealMLP best RMSE: {study_mlp.best_value:.5f}")

    # FT-Transformer with Optuna
    logger.info("\n--- FT-Transformer ---")
    study_ft = __import__("optuna").create_study(direction="minimize", study_name="FT-Transformer")
    study_ft.optimize(
        lambda trial: optimize_ft_transformer(trial, X_train_scaled.values, target.values, cv=cv),
        n_trials=max(10, n_trials // 3),
    )
    with mlflow.start_run(run_name="FTTransformer_optuna"):
        mlflow.log_params(study_ft.best_params)
        mlflow.log_metric("cv_rmse", study_ft.best_value)
    results["FT-Transformer"] = {
        "score": study_ft.best_value,
        "params": study_ft.best_params,
        "uses_scaled": True,
    }
    logger.info(f"FT-Transformer best RMSE: {study_ft.best_value:.5f}")

    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["score"])
    for rank, (name, info) in enumerate(sorted_results, 1):
        logger.info(f"  #{rank} {name}: RMSE = {info['score']:.5f}")

    return results


def build_ensemble(config, data, results, top_n=5):
    """Build a weighted ensemble from top N models."""
    target = data["target"]
    X_train = data["X_train"]
    X_train_scaled = data["X_train_scaled"]
    X_test = data["X_test"]
    X_test_scaled = data["X_test_scaled"]

    # Select top N models
    sorted_models = sorted(results.items(), key=lambda x: x[1]["score"])[:top_n]
    logger.info(f"\nBuilding ensemble from top {len(sorted_models)} models:")
    for name, info in sorted_models:
        logger.info(f"  - {name}: RMSE = {info['score']:.5f}")

    # Build each model and get OOF + test predictions
    kf = KFold(n_splits=config["model"]["cv_folds"], shuffle=True, random_state=42)
    oof_predictions = {}
    test_predictions = {}

    for name, info in sorted_models:
        model = _create_model(name, info["params"])
        X_tr = X_train_scaled if info["uses_scaled"] else X_train
        X_te = X_test_scaled if info["uses_scaled"] else X_test

        # Convert to numpy for deep models
        if name in ("RealMLP", "FT-Transformer"):
            X_tr_vals = X_tr.values
            X_te_vals = X_te.values
        else:
            X_tr_vals = X_tr
            X_te_vals = X_te

        oof = np.zeros(len(target))
        test_preds = np.zeros(len(X_te))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_vals)):
            if isinstance(X_tr_vals, pd.DataFrame):
                X_fold_train = X_tr_vals.iloc[train_idx]
                X_fold_val = X_tr_vals.iloc[val_idx]
            else:
                X_fold_train = X_tr_vals[train_idx]
                X_fold_val = X_tr_vals[val_idx]

            y_fold_train = target.iloc[train_idx]

            model_fold = _create_model(name, info["params"])
            model_fold.fit(X_fold_train, y_fold_train)
            oof[val_idx] = model_fold.predict(X_fold_val)
            test_preds += model_fold.predict(X_te_vals) / kf.n_splits

        oof_rmse = np.sqrt(np.mean((oof - target.values) ** 2))
        logger.info(f"  {name} OOF RMSE: {oof_rmse:.5f}")
        oof_predictions[name] = oof
        test_predictions[name] = test_preds

    # Weighted average (inverse RMSE weighting)
    weights = {}
    total_inv = 0
    for name in oof_predictions:
        rmse = np.sqrt(np.mean((oof_predictions[name] - target.values) ** 2))
        inv_rmse = 1.0 / rmse
        weights[name] = inv_rmse
        total_inv += inv_rmse

    for name in weights:
        weights[name] /= total_inv
        logger.info(f"  {name} weight: {weights[name]:.4f}")

    # Ensemble predictions
    ensemble_oof = sum(weights[name] * oof_predictions[name] for name in weights)
    ensemble_test = sum(weights[name] * test_predictions[name] for name in weights)

    ensemble_rmse = np.sqrt(np.mean((ensemble_oof - target.values) ** 2))
    logger.info(f"\n  Ensemble OOF RMSE: {ensemble_rmse:.5f}")

    return ensemble_test, weights, ensemble_rmse


def _create_model(name, params):
    """Instantiate a model from name and params."""
    params = dict(params)  # copy

    if name == "Ridge":
        return Ridge(**params)
    elif name == "Lasso":
        return Lasso(**params, max_iter=10000)
    elif name == "ElasticNet":
        return ElasticNet(**params, max_iter=10000)
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, random_state=42, verbosity=0)
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, random_state=42, verbosity=-1)
    elif name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params, random_state=42, verbose=0)
    elif name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params, random_state=42)
    elif name == "RealMLP":
        from house_prices.models.deep_models import RealMLPRegressor
        # d_layers comes as tuple from optuna
        if "d_layers" in params and isinstance(params["d_layers"], list):
            params["d_layers"] = tuple(params["d_layers"])
        return RealMLPRegressor(**params, n_epochs=100, patience=15, random_state=42)
    elif name == "FT-Transformer":
        from house_prices.models.deep_models import FTTransformerRegressor
        p = dict(params)
        if p.get("d_token", 64) % p.get("n_heads", 4) != 0:
            p["n_heads"] = min(p["n_heads"], p["d_token"])
            while p["d_token"] % p["n_heads"] != 0:
                p["n_heads"] -= 1
        return FTTransformerRegressor(**p, n_epochs=100, patience=15, random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")


def generate_submission(test_predictions, test_ids, config, log_transform=True):
    """Generate submission CSV."""
    predictions = test_predictions.copy()
    if log_transform:
        predictions = np.expm1(predictions)
    predictions = np.clip(predictions, 0, None)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    submissions_dir = PROJECT_ROOT / config["output"]["submissions_dir"]
    submissions_dir.mkdir(exist_ok=True)
    output_path = submissions_dir / "submission_advanced.csv"
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved: {output_path}")
    logger.info(f"Price range: {predictions.min():.0f} - {predictions.max():.0f}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per model")
    parser.add_argument("--top-n", type=int, default=5, help="Top N models for ensemble")
    args = parser.parse_args()

    config = load_config()

    # 1. Prepare data
    logger.info("Preparing data...")
    data = prepare_pipeline_data(config)
    logger.info(f"Features: {len(data['feature_names'])}")

    # 2. Train all models with Optuna
    results = train_all_models(config, data, n_trials=args.n_trials)

    # 3. Build ensemble
    ensemble_preds, weights, ensemble_rmse = build_ensemble(
        config, data, results, top_n=args.top_n
    )

    # 4. Generate submission
    output_path = generate_submission(
        ensemble_preds,
        data["test_ids"],
        config,
        log_transform=config["preprocessing"]["target"]["log_transform"],
    )

    print(f"\nDone! Ensemble RMSE: {ensemble_rmse:.5f}")
    print(f"Submission: {output_path}")
