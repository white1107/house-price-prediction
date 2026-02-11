"""Data loading utilities."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files from raw data directory."""
    raw_path = Path(raw_dir)
    train = pd.read_csv(raw_path / "train.csv")
    test = pd.read_csv(raw_path / "test.csv")
    logger.info(f"Loaded train: {train.shape}, test: {test.shape}")
    return train, test


def prepare_data(
    train: pd.DataFrame, test: pd.DataFrame, log_transform: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Separate target, optionally log-transform, and combine train+test.

    Returns:
        combined: Combined DataFrame without target
        target: Target series (log-transformed if specified)
        test_ids: Test set IDs for submission
    """
    # Remove outliers: GrLivArea > 4000
    outlier_mask = train["GrLivArea"] > 4000
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        train = train[~outlier_mask].reset_index(drop=True)
        logger.info(f"Removed {n_outliers} outliers (GrLivArea > 4000)")

    # Separate target
    target = train["SalePrice"]
    if log_transform:
        target = np.log1p(target)
        logger.info("Applied log1p transform to target")

    # Store test IDs
    test_ids = test["Id"]

    # Drop Id and target
    train = train.drop(columns=["Id", "SalePrice"])
    test = test.drop(columns=["Id"])

    # Combine for consistent preprocessing
    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    logger.info(f"Combined data shape: {combined.shape} (train: {n_train})")

    return combined, target, test_ids
