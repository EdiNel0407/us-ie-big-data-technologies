# postblock1/q6/Dockerfile
FROM python:3.10-slim

# --- Basics ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Python deps (CPU-only) ---
# 1) Torch CPU build
RUN pip install --upgrade pip \
 && pip install "torch==2.4.1+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html

# 2) ML + GCS client + data utils
RUN pip install \
      "transformers==4.44.2" \
      "accelerate>=0.33.0" \
      "google-cloud-storage>=2.16.0" \
      "pandas>=2.2.2" \
      "pyarrow>=14.0.2" \
      "tqdm>=4.66.4" \
      "numpy<2.0"  # keep numpy <2 to avoid ABI surprises with some wheels

# --- App code ---
WORKDIR /app
COPY bdt-sentiment.py /app/bdt-sentiment.py

# Where the app expects the service-account key (mount one at runtime)
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-key.json

# Optional: set a default input (can be overridden by --input or by changing the env var)
ENV INPUT=gs://YOUR_BUCKET/tweets-sentiment-synth.csv

# Run the script by default; allow overriding input via args
ENTRYPOINT ["python", "/app/bdt-sentiment.py"]
CMD ["--input", "${INPUT}"]
