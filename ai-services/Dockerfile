FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD ML MODELS during build (cached, fast startup)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')" || true
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone'); AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')" || true
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('kk08/CryptoBERT'); AutoModelForSequenceClassification.from_pretrained('kk08/CryptoBERT')" || true

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/models /app/data /app/logs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

