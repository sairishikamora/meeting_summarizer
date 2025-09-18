#!/usr/bin/env python3
"""
evaluate_whisper.py
Transcribe a WAV file with Whisper and compute WER against a ground-truth text file.
"""

import argparse
import os
import string
import whisper
import jiwer

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, trim extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper transcription WER")
    parser.add_argument("--model", default="base.en", help="Whisper model size (tiny.en, base.en, etc.)")
    parser.add_argument("--wav", required=True, help="Path to WAV file")
    parser.add_argument("--gt", required=True, help="Path to ground truth text file")
    args = parser.parse_args()

    if not os.path.isfile(args.wav):
        print(f"❌ WAV file not found: {args.wav}")
        return
    if not os.path.isfile(args.gt):
        print(f"❌ Ground truth file not found: {args.gt}")
        return

    try:
        with open(args.gt, "r", encoding="utf-8-sig") as f:
            gt_text = f.read().strip()
    except Exception as e:
        print(f"❌ Error reading ground truth: {e}")
        return

    print(f"🔄 Loading Whisper model '{args.model}'...")
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"🎤 Transcribing '{args.wav}'...")
    hyp_text = model.transcribe(args.wav, language='en')["text"]

    gt_norm = normalize_text(gt_text)
    hyp_norm = normalize_text(hyp_text)
    wer_score = jiwer.wer(gt_norm, hyp_norm)

    print("\n--- Evaluation Results ---")
    print("GROUND TRUTH (raw):\n", gt_text, "\n")
    print("HYPOTHESIS (raw):\n", hyp_text, "\n")
    print("GROUND TRUTH (normalized):\n", gt_norm, "\n")
    print("HYPOTHESIS (normalized):\n", hyp_norm, "\n")
    print(f"✅ WER = {wer_score*100:.2f}%\n")

if __name__ == "__main__":
    main()
