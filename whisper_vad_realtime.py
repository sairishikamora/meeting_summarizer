import os
import whisper
import torch
import sounddevice as sd
import numpy as np
import time

# --- Configuration ---
WHISPER_MODEL_SIZE = "base.en"
SILENCE_BLOCKS = 3
BLOCK_SIZE = 512
SAMPLE_RATE = 16000

# --- Load VAD model ---
try:
    print("🔹 Loading Silero VAD model...")
    VAD_MODEL, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
except Exception:
    from silero_vad import load_silero_vad
    VAD_MODEL, utils = load_silero_vad()

VAD_MODEL.eval()

def is_speech(chunk, model, sample_rate):
    with torch.no_grad():
        wav_tensor = torch.from_numpy(chunk).float()
        speech_prob = model(wav_tensor, sample_rate).item()
        return speech_prob > 0.5

def main():
    print(f"🔹 Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    
    print("\n🎤 Listening for speech... (Press Ctrl+C to stop)")
    audio_buffer = np.array([], dtype=np.int16)
    silent_blocks_count = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer, silent_blocks_count
        float_data = indata.flatten().astype(np.float32) / 32768.0
        speaking = is_speech(float_data, VAD_MODEL, SAMPLE_RATE)

        if speaking:
            silent_blocks_count = 0
            audio_buffer = np.append(audio_buffer, indata.flatten())
        else:
            silent_blocks_count += 1

        if silent_blocks_count >= SILENCE_BLOCKS and len(audio_buffer) > 0:
            print("\n⏳ Transcribing utterance...")
            try:
                buffer_tensor = torch.from_numpy(audio_buffer).float() / 32768.0
                result = whisper_model.transcribe(buffer_tensor, language='en')
                output_text = result.get("text", "").strip()
                if output_text:
                    print("✅ Transcribed:", output_text)
            except Exception as e:
                print(f"❌ Transcription error: {e}")
            
            audio_buffer = np.array([], dtype=np.int16)
            silent_blocks_count = 0
            print("\n🎤 Listening for next utterance...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=BLOCK_SIZE, callback=audio_callback):
        print("Press Ctrl+C to exit.")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()