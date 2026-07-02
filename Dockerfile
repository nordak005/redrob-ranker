FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TRANSFORMERS_OFFLINE=0 \
    APP=streamlit

# Install only curl (needed for healthcheck). No build-essential — all packages have prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (avoids pulling CUDA variant from PyPI)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the SentenceTransformer model during build
# (models/ is in .dockerignore so COPY . . won't overwrite this cached download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application files
COPY . .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Health check — extended start-period for model + embedding cold start
# start-period=180s: gives the app 3 minutes to load before HF marks it unhealthy
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Launch Streamlit, respecting the PORT env var injected by HF Spaces (default 7860)
CMD ["sh", "-c", "streamlit run app.py \
    --server.port=${PORT:-7860} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false"]