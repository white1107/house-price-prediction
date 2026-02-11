FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY configs/ configs/
COPY src/ src/
COPY data/raw/ data/raw/

CMD ["python", "-m", "house_prices.models.train"]
