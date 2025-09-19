# Small CPU-only base
FROM python:3.11-slim

# Keep it quiet & deterministic
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1

# System deps (just enough for pandas+gcsfs+ssl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      pandas \
      gcsfs fsspec \
      transformers==4.41.0 \
      tqdm && \
    pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.0+cpu torchvision==0.18.0+cpu

# App files
WORKDIR /app
COPY bdt-sentiment.py /app/bdt-sentiment.py

# By default, allow passing INPUT via env or args
ENTRYPOINT ["python", "/app/bdt-sentiment.py"]
# Example default (can be overridden at docker run):
# CMD ["--input", "gs://my-bucket/tweets.csv", "--text-col", "text", "--label-col", "label"]
