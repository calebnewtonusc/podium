FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (API server only, not full training stack)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    redis httpx aiohttp \
    kaggle \
    chromadb sentence-transformers \
    pandas numpy scikit-learn lightgbm scipy xgboost catboost optuna \
    python-dotenv pydantic loguru typer rich \
    docker

# Copy application code
COPY agents/ ./agents/
COPY knowledge/ ./knowledge/
COPY validation/ ./validation/
COPY competition_types/ ./competition_types/
COPY deploy/serve.py ./serve.py

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
