import pandas as pd

def get_wer_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        valid_wers = []

        for wer_value in df['wer']:
            try:
                wer_float = float(str(wer_value).strip('%'))
                valid_wers.append(wer_float)
            except (ValueError, AttributeError):
                pass  # skip invalid values like "ERROR"

        if not valid_wers:
            print(f"❌ No valid WER values found in '{csv_file}'.")
            return

        average_wer = sum(valid_wers) / len(valid_wers)
        print(f"\n✅ Overall Whisper WER: {average_wer:.2f}%\n")

    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    CSV_FILE = "whisper_evaluation_results.csv"
    get_wer_from_csv(CSV_FILE)
