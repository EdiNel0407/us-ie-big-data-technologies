
FROM python:3.10-slim

# Small system deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch + deps (keep image small)
RUN pip install --no-cache-dir "torch==2.2.2+cpu" -f https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==4.44.2 pandas

# App
COPY bdt-sentiment.py /app/bdt-sentiment.py

# Container runs the app; pass --input at runtime
ENTRYPOINT ["python", "/app/bdt-sentiment.py"]
CMD ["--help"]
