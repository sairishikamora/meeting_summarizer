#!/usr/bin/env python3
"""
Diarize an audio file with Pyannote and transcribe each speaker using Whisper.
"""

import os
import argparse
import whisper
import torch
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline

# --- Config ---
WHISPER_MODEL_SIZE = "base.en"
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HUGGING_FACE_TOKEN:
    print("❌ Hugging Face token not found. Set HUGGING_FACE_HUB_TOKEN.")
    exit()

def diarize_and_transcribe(audio_path, whisper_model_size):
    print("🔹 Loading Pyannote diarization model...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGING_FACE_TOKEN
    )

    print(f"🔹 Running diarization on {audio_path}...")
    diarization_result = diarization_pipeline(audio_path)

    print(f"🔹 Loading Whisper model '{whisper_model_size}'...")
    whisper_model = whisper.load_model(whisper_model_size)

    full_audio, _ = sf.read(audio_path)
    transcriptions = []

    print("🔹 Transcribing speaker segments...")
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start_sample = int(turn.start * 16000)
        end_sample = int(turn.end * 16000)
        segment_audio = full_audio[start_sample:end_sample]

        segment_text = whisper_model.transcribe(segment_audio.astype(np.float32))["text"].strip()
        transcriptions.append(f"[{speaker}]: {segment_text}")

    # Save RTTM file
    rttm_path = "output.rttm"
    with open(rttm_path, "w") as f:
        diarization_result.write_rttm(f)
    print(f"✅ Diarization saved to {rttm_path}")

    # Save full transcript
    transcript_path = "diarized_transcript.txt"
    full_transcript = "\n".join(transcriptions)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(full_transcript)
    print(f"✅ Full transcript saved to {transcript_path}")

    return full_transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarize & transcribe an audio file")
    parser.add_argument("--audio", required=True, help="Path to audio file (16kHz mono WAV)")
    parser.add_argument("--model", default=WHISPER_MODEL_SIZE, help="Whisper model size")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        exit()
    info = sf.info(args.audio)
    if info.samplerate != 16000 or info.channels != 1:
        print("❌ Audio must be 16kHz mono WAV.")
        exit()

    final_transcript = diarize_and_transcribe(args.audio, args.model)
    print("\n--- Final Transcript ---")
    print(final_transcript)
