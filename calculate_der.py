#!/usr/bin/env python3
"""
Compare reference and hypothesis RTTM files and calculate Diarization Error Rate (DER).
"""

import os
import argparse
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation

def calculate_der(reference_path, hypothesis_path):
    reference = Annotation.from_file(reference_path)
    hypothesis = Annotation.from_file(hypothesis_path)
    der_metric = DiarizationErrorRate()
    der_score = der_metric(reference, hypothesis)
    return der_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Diarization Error Rate (DER).")
    parser.add_argument("--reference", required=True, help="Ground truth RTTM file")
    parser.add_argument("--hypothesis", required=True, help="Hypothesis RTTM file")
    args = parser.parse_args()

    if not os.path.isfile(args.reference):
        print(f"❌ Reference file not found: {args.reference}")
    elif not os.path.isfile(args.hypothesis):
        print(f"❌ Hypothesis file not found: {args.hypothesis}")
    else:
        der = calculate_der(args.reference, args.hypothesis)
        print("\n--- Diarization Evaluation ---")
        print(f"✅ DER: {der*100:.2f}% (Target < 20%)")
