import sounddevice as sd
import soundfile as sf

# --- Settings ---
SAMPLERATE = 16000    # 16 kHz
DURATION = 5          # seconds
OUTPUT_FILE = "data/test_16k_mono.wav"

print(f"🎤 Recording for {DURATION} seconds... Speak now!")
recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
sd.wait()  # wait until recording is done
sf.write(OUTPUT_FILE, recording, SAMPLERATE, subtype='PCM_16')
print(f"✅ Recording saved to: {OUTPUT_FILE}")
