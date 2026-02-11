"""Generate predictions and submission file."""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


def generate_submission(config: dict) -> Path:
    """Load trained model and generate submission CSV."""
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]

    # Load artifacts
    model = joblib.load(models_dir / "best_model.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    feature_names = joblib.load(models_dir / "feature_names.joblib")
    test_ids = joblib.load(models_dir / "test_ids.joblib")

    logger.info(f"Loaded model: {type(model).__name__}")

    # Load and prepare data
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    train_df, test_df = load_raw_data(str(raw_dir))
    combined, _, _ = prepare_data(
        train_df, test_df, log_transform=config["preprocessing"]["target"]["log_transform"]
    )

    # Feature engineering
    combined = build_features(combined)

    # Get test data
    n_train = len(train_df) - (train_df["GrLivArea"] > 4000).sum()
    X_test = combined.iloc[n_train:]

    # Align columns
    missing_cols = set(feature_names) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[feature_names]

    # Scale and predict
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )
    predictions = model.predict(X_test_scaled)

    # Reverse log transform
    if config["preprocessing"]["target"]["log_transform"]:
        predictions = np.expm1(predictions)

    # Create submission
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    submissions_dir = PROJECT_ROOT / config["output"]["submissions_dir"]
    submissions_dir.mkdir(exist_ok=True)
    output_path = submissions_dir / "submission.csv"
    submission.to_csv(output_path, index=False)

    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Predictions - mean: {predictions.mean():.0f}, median: {np.median(predictions):.0f}")
    return output_path


if __name__ == "__main__":
    config = load_config()
    output_path = generate_submission(config)
    print(f"\nSubmission file: {output_path}")
