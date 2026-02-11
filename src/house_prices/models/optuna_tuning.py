"""Optuna hyperparameter optimization for all models."""

import logging

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def optimize_ridge(trial, X, y, cv=5):
    from sklearn.linear_model import Ridge
    alpha = trial.suggest_float("alpha", 0.1, 100.0, log=True)
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_lasso(trial, X, y, cv=5):
    from sklearn.linear_model import Lasso
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    model = Lasso(alpha=alpha, max_iter=10000)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_elasticnet(trial, X, y, cv=5):
    from sklearn.linear_model import ElasticNet
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_xgboost(trial, X, y, cv=5):
    from xgboost import XGBRegressor
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "random_state": 42,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_catboost(trial, X, y, cv=5):
    from catboost import CatBoostRegressor
    params = {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_state": 42,
        "verbose": 0,
    }
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_lightgbm(trial, X, y, cv=5):
    from lightgbm import LGBMRegressor
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbosity": -1,
    }
    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_gradient_boosting(trial, X, y, cv=5):
    from sklearn.ensemble import GradientBoostingRegressor
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42,
    }
    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


# Mapping of model name -> objective function
OBJECTIVE_MAP = {
    "Ridge": optimize_ridge,
    "Lasso": optimize_lasso,
    "ElasticNet": optimize_elasticnet,
    "XGBoost": optimize_xgboost,
    "CatBoost": optimize_catboost,
    "LightGBM": optimize_lightgbm,
    "GradientBoosting": optimize_gradient_boosting,
}


def run_optuna(
    model_name: str,
    X,
    y,
    n_trials: int = 50,
    cv: int = 5,
) -> dict:
    """Run Optuna optimization for a given model.

    Returns:
        dict with 'best_params', 'best_score', 'study'
    """
    if model_name not in OBJECTIVE_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(OBJECTIVE_MAP.keys())}")

    objective_fn = OBJECTIVE_MAP[model_name]
    study = optuna.create_study(direction="minimize", study_name=model_name)
    study.optimize(lambda trial: objective_fn(trial, X, y, cv=cv), n_trials=n_trials)

    logger.info(f"[{model_name}] Best RMSE: {study.best_value:.5f}")
    logger.info(f"[{model_name}] Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }
