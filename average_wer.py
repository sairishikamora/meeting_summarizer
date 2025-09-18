#!/usr/bin/env python3
"""
Calculate the average WER from a CSV file containing a 'wer' column.
"""

import pandas as pd

def calculate_average_wer(csv_file):
    df = pd.read_csv(csv_file)

    # Convert 'wer' column from string like "2.35%" to float
    df['wer_float'] = df['wer'].str.strip('%').astype(float) / 100

    # Compute the average
    average_wer = df['wer_float'].mean()

    print(f"\n✅ Average WER: {average_wer * 100:.2f}%")

if __name__ == "__main__":
    calculate_average_wer("evaluation_results.csv")
