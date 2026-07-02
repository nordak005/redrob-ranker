FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    APP=streamlit

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies
# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.1

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the SentenceTransformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck for the Streamlit service
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD sh -c 'PORT=${PORT:-8501}; curl -f http://localhost:${PORT}/_stcore/health || exit 1'

# Launch script
CMD ["sh", "-c", "\
    PORT=${PORT:-8501}; \
    if [ \"$APP\" = \"ranker\" ]; then \
        python scripts/generate_submission.py; \
    else \
        streamlit run app.py \
            --server.port=${PORT} \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false; \
    fi"]
