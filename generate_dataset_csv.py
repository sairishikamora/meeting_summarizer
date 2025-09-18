import os
import csv
import argparse
import string

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, trim extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def create_dataset_csv(input_dir, output_csv):
    """Generate CSV with audio paths and transcripts."""
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_path', 'ground_truth', 'normalized_gt'])

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.trans.txt'):
                    trans_path = os.path.join(root, file)
                    print(f"📄 Parsing transcript: {trans_path}")
                    with open(trans_path, 'r', encoding='utf-8') as trans_f:
                        for line in trans_f:
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                file_id, text = parts
                                audio_file = f"{file_id}.flac"
                                audio_path = os.path.join(root, audio_file)
                                normalized_text = normalize_text(text)
                                writer.writerow([audio_path, text, normalized_text])

    print(f"\n✅ Dataset CSV created successfully at {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV from LibriSpeech transcripts")
    parser.add_argument("--input", required=True, help="Path to LibriSpeech dataset root")
    parser.add_argument("--output", default="dataset.csv", help="Output CSV file path")
    args = parser.parse_args()
    create_dataset_csv(args.input, args.output)
