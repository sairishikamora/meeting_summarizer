#!/usr/bin/env python3
"""
Batch convert FLAC files to 16kHz mono WAV using ffmpeg.
"""

import os
import subprocess
import argparse

def convert_flac_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                relative_path = os.path.relpath(flac_path, input_dir)
                wav_path_base = os.path.splitext(relative_path)[0]
                output_wav_path = os.path.join(output_dir, wav_path_base + ".wav")

                output_wav_dir = os.path.dirname(output_wav_path)
                if not os.path.exists(output_wav_dir):
                    os.makedirs(output_wav_dir)

                print(f"🔹 Converting {flac_path} → {output_wav_path}")
                try:
                    subprocess.run([
                        'ffmpeg', '-i', flac_path,
                        '-ac', '1', '-ar', '16000', '-sample_fmt', 's16',
                        '-y', output_wav_path
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error converting {flac_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FLAC to 16kHz mono WAV")
    parser.add_argument("--input", required=True, help="Directory with FLAC files")
    parser.add_argument("--output", required=True, help="Directory for WAV output")
    args = parser.parse_args()

    convert_flac_to_wav(args.input, args.output)
    print("\n✅ Batch conversion complete")
