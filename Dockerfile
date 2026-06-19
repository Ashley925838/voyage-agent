cat > Dockerfile << 'EOF'
# syntax=docker/dockerfile:1.6

# Build a slim CPU-only image. We avoid GPU torch wheels (would be ~2.5 GB).
FROM python:3.11-slim

# System deps: faiss-cpu needs libgomp.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# Install Python deps first for layer caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so cold-start latency is bounded
# (otherwise the first /chat request stalls ~30s downloading bge-small).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy application code.
COPY --chown=user app/ ./app/

# HF Spaces injects PORT=7860 and routes traffic to it.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
EOF