#!/usr/bin/env python3
"""Download and test a Hugging Face text classification model."""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TEXT = "I loved this tutorial. It was practical and easy to follow."
DEFAULT_CACHE_DIR = "hf_cache"
DEFAULT_SAMPLES_FILE = "samples.txt"


def load_samples(samples_path: Path) -> List[Tuple[Optional[str], str]]:
    """Read optional LABEL<TAB>text lines from a sample file."""
    entries: List[Tuple[Optional[str], str]] = []
    for raw_line in samples_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "\t" in line:
            expected, text = line.split("\t", 1)
            expected_label = expected.strip().upper() or None
            sample_text = text.strip()
        else:
            expected_label = None
            sample_text = line

        if sample_text:
            entries.append((expected_label, sample_text))

    if not entries:
        raise ValueError(f"No usable samples found in: {samples_path}")

    return entries


def predict_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str,
) -> List[Tuple[str, float]]:
    """Run batched inference and return (label, confidence) results."""
    encoded = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    results: List[Tuple[str, float]] = []
    for row in probabilities:
        predicted_index = int(torch.argmax(row).item())
        label = model.config.id2label[predicted_index]
        confidence = float(row[predicted_index].item())
        results.append((label, confidence))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text classification with a Hugging Face model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id for sequence classification.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to classify.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Folder (inside the current working directory) to store downloaded model files.",
    )
    parser.add_argument(
        "--samples-file",
        default=None,
        help=(
            "Optional sample file path. Each line should be either 'TEXT' or "
            "'LABEL<TAB>TEXT'."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path.cwd() / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, cache_dir=str(cache_dir)
    )
    model.to(device)
    model.eval()

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Download directory: {cache_dir.resolve()}")

    samples_file = args.samples_file
    if samples_file is None and (Path.cwd() / DEFAULT_SAMPLES_FILE).exists():
        samples_file = DEFAULT_SAMPLES_FILE

    if samples_file is not None:
        samples_path = Path(samples_file)
        if not samples_path.is_absolute():
            samples_path = Path.cwd() / samples_path

        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")

        entries = load_samples(samples_path)
        texts = [text for _, text in entries]
        predictions = predict_texts(model, tokenizer, texts, device)

        print(f"Samples file: {samples_path.resolve()}")
        labeled_total = 0
        labeled_correct = 0

        for index, ((expected, text), (predicted, confidence)) in enumerate(
            zip(entries, predictions), start=1
        ):
            if expected is not None:
                labeled_total += 1
                is_match = expected == predicted.upper()
                if is_match:
                    labeled_correct += 1
                status = "OK" if is_match else "MISS"
                print(
                    f"{index:02d}. expected={expected} predicted={predicted} "
                    f"confidence={confidence:.4f} status={status} text={text}"
                )
            else:
                print(
                    f"{index:02d}. predicted={predicted} "
                    f"confidence={confidence:.4f} text={text}"
                )

        if labeled_total > 0:
            accuracy = labeled_correct / labeled_total
            print(f"Accuracy: {labeled_correct}/{labeled_total} = {accuracy:.4f}")
    else:
        prediction = predict_texts(model, tokenizer, [args.text], device)[0]
        label, confidence = prediction
        print(f"Input: {args.text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
