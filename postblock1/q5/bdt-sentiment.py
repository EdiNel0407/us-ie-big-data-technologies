
import argparse, os
import pandas as pd
from transformers import pipeline

def _norm_label(s: str) -> str:
    s = str(s).strip().lower()
    # normalize common label variants
    mapping = {
        "negative":"neg", "0":"neg", "-1":"neg",
        "neutral":"neu",  "1":"neu",
        "positive":"pos", "2":"pos"
    }
    return mapping.get(s, s)

def main():
    ap = argparse.ArgumentParser(description="Tweet sentiment accuracy with a HF model")
    ap.add_argument("--input","-i", required=True, help="CSV with tweet text + ground-truth column")
    ap.add_argument("--text-col", default="text", help="Column with tweet text")
    ap.add_argument("--label-col", default="label", help="Column with ground-truth label")
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    help="HF model name")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        # fallback common names
        for c in ["tweet", "content", "message"]:
            if c in df.columns:
                args.text_col = c; break
    if args.label_col not in df.columns:
        for c in ["sentiment", "ground_truth", "target"]:
            if c in df.columns:
                args.label_col = c; break

    texts = df[args.text_col].astype(str).tolist()
    y_true = [_norm_label(v) for v in df[args.label_col].tolist()]

    clf = pipeline("sentiment-analysis",
                   model=args.model, tokenizer=args.model, truncation=True)

    # batch inference
    preds = []
    for i in range(0, len(texts), args.batch_size):
        preds.extend(clf(texts[i:i+args.batch_size]))

    y_pred = [_norm_label(p["label"]) for p in preds]
    acc = (pd.Series(y_pred) == pd.Series(y_true)).mean()
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%) on {len(y_true)} tweets")

if __name__ == "__main__":
    main()
