#!/usr/bin/env python3
"""
bdt-sentiment.py

Usage:
  python bdt-sentiment.py --input path/to/tweets.csv
  python bdt-sentiment.py --input gs://YOUR_BUCKET/tweets.csv    # reads directly from GCS

CSV expectations:
- One column with the tweet text (named 'text', 'tweet', 'message', or similar)
- One column with ground-truth sentiment (named 'sentiment', 'label', or 'target')
  Accepted label values (case-insensitive): negative/0, neutral/1, positive/2
"""

import argparse
import io
import os
import re
from typing import List, Tuple

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
import torch


# ---------------------------
# Model: strong tweet-specific checkpoint
# ---------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ---------------------------
# Utilities
# ---------------------------
TEXT_CANDIDATES = ["text", "tweet", "message", "content", "body"]
LABEL_CANDIDATES = ["sentiment", "label", "target", "y", "class"]

LABEL_NORMALISE = {
    "neg": "negative",
    "negative": "negative",
    "0": "negative",
    0: "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "1": "neutral",
    1: "neutral",
    "pos": "positive",
    "positive": "positive",
    "2": "positive",
    2: "positive",
}

url_pat = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
user_pat = re.compile(r"@\w+")
hashtag_pat = re.compile(r"#(\w+)")


def tweet_clean(s: str) -> str:
    """Light normalisation that helps Twitter models a bit."""
    s = url_pat.sub("http", s)
    s = user_pat.sub("@user", s)
    s = hashtag_pat.sub(r"\1", s)  # remove # but keep the token
    return s.strip()


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    # fallback: try fuzzy contains
    for c in df.columns:
        lc = c.lower()
        if any(name in lc for name in candidates):
            return c
    raise ValueError(
        f"Could not find a suitable column among {df.columns.tolist()} for candidates {candidates}"
    )


def read_csv_anywhere(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        # read from Google Cloud Storage using google-cloud-storage
        try:
            from google.cloud import storage  # lazy import
        except Exception as e:
            raise RuntimeError(
                "Reading from GCS requires the 'google-cloud-storage' package and "
                "application credentials. Install with 'pip install google-cloud-storage' "
                "and set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON."
            ) from e

        # parse gs://bucket/obj
        _, _, bucket, *obj_parts = path.split("/")
        blob_name = "/".join(obj_parts)
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        data = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(data))
    else:
        return pd.read_csv(path)


def to_standard_label(x) -> str:
    if isinstance(x, str):
        key = x.strip().lower()
    else:
        key = str(x)
    if key in LABEL_NORMALISE:
        return LABEL_NORMALISE[key]
    # try numeric
    if key.isdigit():
        return LABEL_NORMALISE.get(key, None)
    return None


def evaluate(texts: List[str], gold: List[str]) -> Tuple[float, List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, id2label=ID2LABEL, label2id=LABEL2ID
    )

    device = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        top_k=None,
        batch_size=32,
    )

    preds = pipe(texts)
    # pipeline returns dicts with 'label' and 'score'
    pred_labels = []
    for p in preds:
        # Some versions return 'LABEL_0/1/2' -> map with ID2LABEL if needed
        lab = p.get("label", "")
        lab_l = lab.lower()
        if lab_l in ID2LABEL.values():
            pred_labels.append(lab_l)
        else:
            # handle LABEL_0/1/2
            if lab_l.startswith("label_"):
                idx = int(lab_l.split("_")[-1])
                pred_labels.append(ID2LABEL[idx])
            else:
                # last resort
                pred_labels.append(lab_l)

    correct = sum(1 for a, b in zip(pred_labels, gold) if a == b)
    acc = correct / max(1, len(gold))
    return acc, pred_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV path or gs://bucket/file.csv")
    args = ap.parse_args()

    df = read_csv_anywhere(args.input)

    text_col = find_column(df, TEXT_CANDIDATES)
    label_col = find_column(df, LABEL_CANDIDATES)

    # clean & normalise
    texts = [tweet_clean(str(t)) for t in df[text_col].astype(str).tolist()]
    gold = [to_standard_label(v) for v in df[label_col].tolist()]

    # filter out rows with unknown labels
    keep = [i for i, g in enumerate(gold) if g in {"negative", "neutral", "positive"}]
    if not keep:
        raise ValueError(
            f"No usable labels found in '{label_col}'. Examples seen: "
            f"{pd.Series(df[label_col]).head(5).tolist()}"
        )

    texts = [texts[i] for i in keep]
    gold = [gold[i] for i in keep]

    acc, _ = evaluate(texts, gold)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%) on {len(gold)} tweets")


if __name__ == "__main__":
    main()
