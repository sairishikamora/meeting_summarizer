#!/usr/bin/env python3
"""
evaluate_vosk.py
Transcribe a WAV file with Vosk and compute WER against a ground-truth text file.
"""

import argparse
import os
import wave
import json
import string
from vosk import Model, KaldiRecognizer
from jiwer import wer

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, trim extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def transcribe_wav(model_path, wav_path):
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        print("⚠️ WARNING: WAV should be 16kHz, mono, 16-bit. Consider resampling with ffmpeg:")
        print("   ffmpeg -i input.wav -ac 1 -ar 16000 -sample_fmt s16 output_16k_mono.wav")

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            results.append(r.get("text", ""))

    r = json.loads(rec.FinalResult())
    results.append(r.get("text", ""))
    return " ".join(results).strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Vosk transcription WER")
    parser.add_argument("--model", required=True, help="Path to Vosk model directory")
    parser.add_argument("--wav", required=True, help="Path to WAV file (16kHz mono)")
    parser.add_argument("--gt", required=True, help="Path to ground truth text file")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"❌ Model path not found: {args.model}")
        return
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

    print(f"🎤 Transcribing '{args.wav}' using Vosk model...")
    hyp_text = transcribe_wav(args.model, args.wav)

    gt_norm = normalize_text(gt_text)
    hyp_norm = normalize_text(hyp_text)
    wer_score = wer(gt_norm, hyp_norm)

    print("\n--- Evaluation Results ---")
    print("GROUND TRUTH (raw):\n", gt_text, "\n")
    print("HYPOTHESIS (raw):\n", hyp_text, "\n")
    print("GROUND TRUTH (normalized):\n", gt_norm, "\n")
    print("HYPOTHESIS (normalized):\n", hyp_norm, "\n")
    print(f"✅ WER = {wer_score*100:.2f}%\n")

if __name__ == "__main__":
    main()
