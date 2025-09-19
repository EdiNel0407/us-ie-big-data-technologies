#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---- args
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to CSV with columns: text, label")
args = parser.parse_args()

# ---- load data
df = pd.read_csv(args.input)
# expected columns: text, label (label can be 0/1/2 or negative/neutral/positive)
text_col = "text"
label_col = "label"
assert text_col in df.columns and label_col in df.columns, "CSV must have columns: text, label"

def norm_label(x):
    """Normalize ground-truth labels to {'negative','neutral','positive'}."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    # numeric labels
    if s in {"0", "neg", "negative", "label_0"}:
        return "negative"
    if s in {"1", "neu", "neutral", "label_1"}:
        return "neutral"
    if s in {"2", "pos", "positive", "label_2"}:
        return "positive"
    # fallbacks
    if "neg" in s:
        return "negative"
    if "neu" in s:
        return "neutral"
    if "pos" in s:
        return "positive"
    return None

df["y_true"] = df[label_col].map(norm_label)

# ---- model: 3-class Twitter RoBERTa (great on tweets)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

clf = pipeline(
    "text-classification",
    model=mdl,
    tokenizer=tok,
    return_all_scores=False,
    truncation=True,
    padding=False,
    top_k=1,
    device=-1,          # CPU (compatible with your Docker image)
)

# ---- batched inference
texts = df[text_col].astype(str).tolist()
preds = clf(texts, batch_size=32)

# cardiffnlp returns LABEL_0/1/2; map to words
id2lab = {0: "negative", 1: "neutral", 2: "positive"}
def pipe_label_to_word(item):
    lab = item["label"]
    if isinstance(lab, str) and lab.startswith("LABEL_"):
        try:
            idx = int(lab.split("_")[-1])
            return id2lab.get(idx, "neutral")
        except:
            return "neutral"
    # sometimes models return capitalized words; normalize
    return str(lab).strip().lower()

df["y_pred"] = [pipe_label_to_word(p) for p in preds]

# ---- accuracy
mask_valid = df["y_true"].notna()
n = int(mask_valid.sum())
acc = float((df.loc[mask_valid, "y_true"] == df.loc[mask_valid, "y_pred"]).mean()) if n else 0.0

# per-class stats (simple counts)
def counts(col):
    return Counter(df.loc[mask_valid, col].tolist())

true_counts = counts("y_true")
pred_counts = counts("y_pred")

# confusion-ish counts (only for quick console view)
def conf_count(t, p):
    return int(((df["y_true"] == t) & (df["y_pred"] == p) & mask_valid).sum())

labels = ["negative", "neutral", "positive"]

print("\n================ Sentiment Accuracy Report ================\n")
print(f"Model: {MODEL_NAME}")
print(f"Rows evaluated: {n}")
print(f"Accuracy: {acc*100:.2f}%\n")

print("Class distribution (true):")
for c in labels:
    print(f"  {c:8s}: {true_counts.get(c,0)}")
print("\nClass distribution (pred):")
for c in labels:
    print(f"  {c:8s}: {pred_counts.get(c,0)}")

print("\nConfusion counts (true -> pred):")
for t in labels:
    row = ", ".join([f"{t}->{p}: {conf_count(t,p)}" for p in labels])
    print("  " + row)

print("\n===========================================================\n")
