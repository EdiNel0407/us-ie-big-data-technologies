
import argparse
import csv
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV with columns: text, label")
    args = parser.parse_args()

    clf = pipeline("sentiment-analysis")  # default: distilbert-base-uncased-finetuned-sst-2-english

    total = 0
    correct = 0

    with open(args.input, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            truth = row.get("label", "").strip().lower()
            if not text:
                continue
            pred = clf(text)[0]
            # model labels: POSITIVE / NEGATIVE
            model_label = pred["label"].strip().lower()
            # allow mapping: positive/negative vs. pos/neg
            model_label = "positive" if "pos" in model_label else "negative"

            total += 1
            if truth == model_label:
                correct += 1

    acc = correct / total if total else 0.0
    print(f"Accuracy: {acc:.4f}  ({correct}/{total})")

if __name__ == "__main__":
    main()
