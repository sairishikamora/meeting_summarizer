import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "t5-small"  # Summarization model

def main():
    parser = argparse.ArgumentParser(description="Generate summary from diarized transcript")
    parser.add_argument("--transcript", required=True, help="Path to transcript file")
    args = parser.parse_args()

    INPUT_FILE = args.transcript

    if not os.path.isfile(INPUT_FILE):
        print(f"❌ Transcript file not found: {INPUT_FILE}")
        exit()

    print(f"🔹 Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    processed_text = "summarize: " + transcript_text
    inputs = tokenizer(processed_text, return_tensors="pt", max_length=1024, truncation=True)

    print("⏳ Generating summary...")
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=150,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )

    output_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("\n✅ Generated Summary:")
    print(output_summary)

    output_path = "summary_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_summary)
    print(f"💾 Summary saved to {output_path}")

if __name__ == "__main__":
    main()