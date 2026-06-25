FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies (installed before copying code for layer caching)
# ---------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Expose Streamlit default port
EXPOSE 8501

# Healthcheck for the Streamlit sandbox
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ---------------------------------------------------------------------------
# Entry point
#
# Default: Streamlit sandbox (for HuggingFace Spaces / Streamlit Cloud)
#
# To run the submission generator instead:
#   docker run -e APP=ranker <image>
#
# To generate a submission with local data mounted:
#   docker run -e APP=ranker \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/outputs:/app/outputs \
#     <image>
# ---------------------------------------------------------------------------
ENV APP=streamlit

CMD ["sh", "-c", "\
    if [ \"$APP\" = \"ranker\" ]; then \
        python scripts/generate_submission.py; \
    else \
        streamlit run app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false; \
    fi"]
