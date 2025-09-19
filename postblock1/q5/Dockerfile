# ---- Base: slim Python with no CUDA (CPU-only) ----
FROM python:3.10-slim

# Ensure stdout/stderr are unbuffered (logs appear immediately)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Small, necessary OS deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps (CPU-only torch + transformers + pandas + gcsfs) ----
# Torch CPU builds are served from the official url below
RUN pip install --no-cache-dir \
      "torch==2.4.1" -f https://download.pytorch.org/whl/cpu/torch_stable.html \
      "transformers==4.44.2" \
      "pandas>=2.0,<3.0" \
      "gcsfs>=2024.0.0"

# ---- App ----
WORKDIR /app
COPY bdt-sentiment.py /app/bdt-sentiment.py

# Default command prints help unless you pass --input or mount /data
# We provide a sensible default so a single docker run will execute.
ENTRYPOINT ["python", "/app/bdt-sentiment.py"]
CMD ["--input", "/data/tweets-sentiment-synth.csv"]

