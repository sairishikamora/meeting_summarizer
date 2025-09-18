"""
Whisper Test Script
Author: Rishika Mora
Description:
    This script loads an OpenAI Whisper model
    and performs transcription on a test audio file.
"""

import whisper
import os

# --- Configuration ---
# Available model sizes: tiny, base, small, medium, large
# English-only versions (e.g., base.en) are slightly faster
MODEL_SIZE = "base.en"

# Path to your input audio file (stored inside data/ folder)
INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "test_16k_mono.wav")

def transcribe_audio(file_path: str, model_size: str = MODEL_SIZE):
    """Transcribes the given audio file using Whisper."""
    if not os.path.exists(file_path):
        print(f"⚠️ Audio file not found: {file_path}")
        print("Please ensure 'test_16k_mono.wav' is placed in the data/ folder.")
        return None
    
    print(f"🔹 Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"🎵 Transcribing input file: {file_path}")
    output_text = model.transcribe(file_path)

    print("\n--- Transcription Result ---")
    print(f"📝 Text: {output_text['text']}")
    print(f"🌍 Language: {output_text['language']}")
    return output_text

# Run the function
if __name__ == "__main__":
    transcribe_audio(INPUT_FILE)
