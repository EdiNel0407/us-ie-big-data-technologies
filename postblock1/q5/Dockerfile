
FROM python:3.10-slim

# Avoid interactive prompts and set UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Basic OS deps that help wheels install cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only transformers + torch (no CUDA)
RUN pip install --upgrade pip \
 && pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers==4.44.2

# Add app
WORKDIR /app
COPY bdt-sentiment.py /app/bdt-sentiment.py

# Default command prints help; override with -e INPUT=... and a volume mount
CMD ["python", "bdt-sentiment.py", "--input", "/data/tweets-sentiment-synth.csv"]
