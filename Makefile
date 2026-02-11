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

train-advanced:
	cd src && python -m house_prices.models.train_advanced --n-trials 50 --top-n 5

train-quick:
	cd src && python -m house_prices.models.train_advanced --n-trials 10 --top-n 3

predict:
	python -m house_prices.models.predict

submit: train predict
	@echo "Submission file created in submissions/"

# =============================================
# API
# =============================================
api:
	PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-docker:
	docker compose up --build

app:
	PYTHONPATH=src streamlit run app.py

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

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
