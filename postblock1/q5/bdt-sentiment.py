#!/usr/bin/env python3
"""
bdt-sentiment.py
- Loads a CSV of tweets from Google Cloud Storage (gs://...) or local disk
- Runs a pretrained Hugging Face transformer for sentiment (no training)
- Compares predictions to ground-truth labels and prints accuracy

Run:
  python bdt-sentiment.py --input gs://<bucket>/<path>.csv --text-col text --label-col label

Assumptions:
- CSV has at least two columns: one with tweet text, one with ground-truth label.
- Labels may be 'positive'/'negative' (any case) OR 1/0. We'll normalize them.

Tip (GCS auth):
- Locally: `gcloud auth application-default login`
- Docker: mount a service-account JSON and export GOOGLE_APPLICATION_CREDENTIALS
"""

import argparse
import os
import sys
from typing import List, Tuple

import pandas as pd

# transformers prints a lot; keep it tidy
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from transformers import pipeline  # noqa: E402


def normalize_truth(series: pd.Series) -> pd.Series:
    """Map ground truth to {'positive','negative'}."""
    def _norm(v):
        if pd.isna(v):
            return None
        s = str(v).strip().lower()
        if s in {"1", "pos", "positive", "true", "t"}:
            return "positive"
        if s in {"0", "neg", "negative", "false", "f"}:
            return "negative"
        # anything else? try a simple heuristic:
        return "positive" if "pos" in s else ("negative" if "neg" in s else s)
    return series.map(_norm)


def normalize_pred(label: str) -> str:
    """Map model labels to {'positive','negative'}."""
    l = label.strip().upper()
    if l == "POSITIVE":
        return "positive"
    if l == "NEGATIVE":
        return "negative"
    return l.lower()


def batched(iterable: List[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def run_eval(
    input_path: str,
    text_col: str,
    label_col: str,
    model_name: str,
    batch_size: int,
    max_rows: int = None,
) -> Tuple[float, int, int, int, int]:
    """
    Return (accuracy, tp, tn, fp, fn).
    """
    # pandas + gcsfs lets us read gs:// paths transparently (when gcsfs installed)
    df = pd.read_csv(input_path)
    if max_rows:
        df = df.head(max_rows)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must have columns '{text_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )

    texts = df[text_col].astype(str).tolist()
    truth = normalize_truth(df[label_col])

    # load HF pipeline (pretrained, CPU)
    clf = pipeline("sentiment-analysis", model=model_name, device=-1)

    preds: List[str] = []
    for chunk in batched(texts, batch_size):
        out = clf(chunk, truncation=True)  # list of dicts: [{'label': 'POSITIVE', 'score': 0.99}, ...]
        preds.extend([normalize_pred(o["label"]) for o in out])

    # compute metrics
    tp = tn = fp = fn = 0
    correct = 0
    total = 0
    for y_true, y_pred in zip(truth, preds):
        if y_true not in {"positive", "negative"}:
            # skip weird/missing ground-truth rows
            continue
        total += 1
        if y_pred == y_true:
            correct += 1
            if y_true == "positive":
                tp += 1
            else:
                tn += 1
        else:
            if y_pred == "positive":
                fp += 1
            else:
                fn += 1

    acc = (correct / total) if total else 0.0
    return acc, tp, tn, fp, fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained tweet sentiment vs ground truth.")
    parser.add_argument("--input", required=True, help="CSV path (gs://bucket/file.csv or local path)")
    parser.add_argument("--text-col", default="text", help="Tweet text column name")
    parser.add_argument("--label-col", default="label", help="Ground truth column name")
    parser.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english",
                        help="HF model name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap to test quickly")
    args = parser.parse_args()

    try:
        acc, tp, tn, fp, fn = run_eval(
            input_path=args.input,
            text_col=args.text_col,
            label_col=args.label-col if hasattr(args, "label-col") else args.label_col,  # guard for shell hyphen issues
            model_name=args.model,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    print("\n=== Sentiment Accuracy Report ===")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Text column: {args.text_col} | Label column: {args.label_col}")
    print(f"Batch size: {args.batch_size} | Max rows: {args.max_rows}")
    print("---------------------------------")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion (pos = positive, neg = negative):")
    print(f"  TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
    print("=================================\n")


if __name__ == "__main__":
    # allow env var as a convenience when using Docker CMD only
    if len(sys.argv) == 1 and os.getenv("INPUT"):
        sys.argv += ["--input", os.environ["INPUT"]]
        if os.getenv("TEXT_COL"):
            sys.argv += ["--text-col", os.environ["TEXT_COL"]]
        if os.getenv("LABEL_COL"):
            sys.argv += ["--label-col", os.environ["LABEL_COL"]]
        if os.getenv("MODEL"):
            sys.argv += ["--model", os.environ["MODEL"]]
        if os.getenv("BATCH_SIZE"):
            sys.argv += ["--batch-size", os.environ["BATCH_SIZE"]]

    main()
