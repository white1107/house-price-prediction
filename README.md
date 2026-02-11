# House Prices: Advanced Regression Techniques

Kaggle competition solution with production-grade ML pipeline and MLOps practices.

**Competition:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Project Structure

```
house-price-prediction/
├── configs/
│   └── config.yaml          # Centralized pipeline configuration
├── data/
│   ├── raw/                  # Original competition data
│   └── processed/            # Preprocessed artifacts
├── models/                   # Trained model artifacts (.joblib)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_preprocessing_and_model.ipynb  # Interactive pipeline
├── src/house_prices/
│   ├── data/loader.py        # Data loading & preparation
│   ├── features/engineering.py  # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py          # Training with MLflow tracking
│   │   └── predict.py        # Inference & submission generation
│   └── utils/config.py       # Configuration management
├── submissions/              # Kaggle submission CSVs
├── tests/                    # Unit tests (pytest)
├── .github/workflows/ci.yml  # CI pipeline
├── Dockerfile                # Containerized training
├── Makefile                  # Task automation
└── pyproject.toml            # Dependencies & tooling config
```

## Quick Start

```bash
# Setup
make setup

# Download data (requires Kaggle API credentials)
make data

# Train models (evaluates Ridge, Lasso, ElasticNet, RF, GBT, XGBoost, LightGBM)
make train

# Generate submission
make predict

# Or run full pipeline
make submit
```

## ML Pipeline

### Data Preprocessing
- Domain-aware missing value imputation (NA = "no feature" vs true missing)
- Neighborhood-grouped LotFrontage imputation
- Outlier removal (GrLivArea > 4000 sqft)

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `TotalSF` | Total square footage (basement + 1st + 2nd floor) |
| `TotalBathrooms` | Combined full + half bathrooms |
| `TotalPorchSF` | Combined porch area |
| `HouseAge` | Age at time of sale |
| `RemodAge` | Years since remodel |
| `HasPool/Garage/Fireplace` | Binary feature flags |

### Encoding Strategy
- **Ordinal features** (quality ratings): Mapped to numeric scale (Ex=5 → Po=1)
- **Nominal features**: One-hot encoded with `drop_first=True`

### Models Evaluated
| Model | Type | Key Hyperparameters |
|-------|------|-------------------|
| Ridge | Linear (L2) | alpha=10.0 |
| Lasso | Linear (L1) | alpha=0.0005 |
| ElasticNet | Linear (L1+L2) | alpha=0.0005, l1_ratio=0.5 |
| Random Forest | Ensemble (Bagging) | n_estimators=300 |
| Gradient Boosting | Ensemble (Boosting) | n_estimators=300, lr=0.05 |
| XGBoost | Ensemble (Boosting) | n_estimators=500, lr=0.05 |
| LightGBM | Ensemble (Boosting) | n_estimators=500, lr=0.05 |

## MLOps Features

- **Experiment Tracking**: MLflow tracks all runs, parameters, and metrics
- **Configuration Management**: YAML-based config (`configs/config.yaml`)
- **Reproducibility**: Fixed random seeds, pinned dependencies
- **Testing**: pytest with coverage reporting
- **CI/CD**: GitHub Actions (lint + test on Python 3.10/3.11/3.12)
- **Containerization**: Dockerfile for portable training
- **Task Automation**: Makefile for common operations

## Development

```bash
# Run tests
make test

# Lint
make lint

# Format code
make format

# View MLflow UI
mlflow ui --backend-store-uri mlruns
```

## Tech Stack

Python | scikit-learn | XGBoost | LightGBM | MLflow | pandas | pytest | GitHub Actions | Docker
