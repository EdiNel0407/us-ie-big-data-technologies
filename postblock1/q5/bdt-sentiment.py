#!/usr/bin/env python3
"""
bdt-sentiment.py
- Reads a CSV with columns: text, sentiment
- If --input starts with gs:// it reads straight from Google Cloud Storage
- Runs a Hugging Face sentiment model and prints accuracy
"""

import argparse
import io
import re
import sys
from typing import List

import pandas as pd
from transformers import pipeline
from google.cloud import storage  # <-- NEW: GCS client


# ---------------------------
# Helpers
# ---------------------------
def _read_csv_from_gcs(gs_url: str) -> pd.DataFrame:
    """
    Read CSV directly from GCS 
    
    """
    if not gs_url.startswith("gs://"):
        raise ValueError(f"Expected gs:// URL, got: {gs_url}")

    m = re.match(r"^gs://([^/]+)/(.+)$", gs_url)
    if not m:
        raise ValueError(f"Malformed GCS URL: {gs_url}")
    bucket_name, blob_name = m.group(1), m.group(2)

    client = storage.Client()                    # Uses ADC (env var or workload identity)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    csv_bytes = blob.download_as_bytes()         # in-memory
    return pd.read_csv(io.BytesIO(csv_bytes))    # If your CSV is UTF-8, this is perfect


def _load_dataframe(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        return _read_csv_from_gcs(path)
    return pd.read_csv(path)


def _normalize_truth(labels: pd.Series) -> List[str]:
    """
    Normalize ground-truth strings to one of: negative / neutral / positive
    """
    norm = (
        labels.astype(str)
        .str.strip()
        .str.lower()
        .replace({"neg": "negative", "pos": "positive"})
    )
    # anything not in the known set becomes neutral (optional)
    norm = norm.where(norm.isin({"negative", "neutral", "positive"}), other="neutral")
    return norm.tolist()


def _normalize_pred(hf_labels: List[str]) -> List[str]:
    """
    Map model outputs to the same three classes.
    Works with 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    which returns labels like 'negative', 'neutral', 'positive'
    or 'LABEL_0/1/2' depending on version.
    """
    out = []
    for lab in hf_labels:
        l = lab.strip().lower()
        if l in {"negative", "neutral", "positive"}:
            out.append(l)
        elif l in {"label_0", "0"}:
            out.append("negative")
        elif l in {"label_1", "1"}:
            out.append("neutral")
        elif l in {"label_2", "2"}:
            out.append("positive")
        else:
            out.append("neutral")
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV (local or gs://bucket/path.csv)")
    ap.add_argument("--text_col", default="text", help="Name of text column")
    ap.add_argument("--label_col", default="sentiment", help="Name of ground-truth column")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # 1) Load data (local or GCS)
    df = _load_dataframe(args.input)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        print(f"Expected columns '{args.text_col}' and '{args.label_col}' in CSV.", file=sys.stderr)
        sys.exit(2)

    texts = df[args.text_col].astype(str).tolist()
    truth = _normalize_truth(df[args.label_col])

    # 2) Build pipeline (CPU)
    clf = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,  # CPU
        truncation=True
    )

    # 3) Run inference (batched)
    preds = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        out = clf(batch)
        preds.extend([o["label"] for o in out])

    # 4) Normalize predictions and compute accuracy
    pred_norm = _normalize_pred(preds)
    correct = sum(1 for a, b in zip(pred_norm, truth) if a == b)
    n = len(truth)
    acc = correct / n if n else 0.0

    print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%) on {n} tweets")


if __name__ == "__main__":
    main()
