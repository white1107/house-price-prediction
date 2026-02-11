.PHONY: setup train predict submit test lint clean eda

# =============================================
# Environment
# =============================================
setup:
	pip install -e ".[dev]"
	pre-commit install

# =============================================
# ML Pipeline
# =============================================
train:
	python -m house_prices.models.train

predict:
	python -m house_prices.models.predict

submit: train predict
	@echo "Submission file created in submissions/"

# =============================================
# EDA
# =============================================
eda:
	jupyter notebook notebooks/01_eda.ipynb

# =============================================
# Quality
# =============================================
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# =============================================
# Data
# =============================================
data:
	kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw/
	cd data/raw && unzip -o house-prices-advanced-regression-techniques.zip

# =============================================
# Cleanup
# =============================================
clean:
	rm -rf __pycache__ .pytest_cache mlruns/ dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
