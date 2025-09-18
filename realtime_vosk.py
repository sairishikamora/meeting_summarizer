import argparse
import os
import queue
import sys
import json
import datetime
import sounddevice as sd
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️ Status:", status, file=sys.stderr)
    q.put(bytes(indata))

def list_devices():
    print(sd.query_devices())

def main():
    parser = argparse.ArgumentParser(description="Realtime STT with Vosk")
    parser.add_argument("--model", required=True, help="Path to Vosk model folder")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate (default 16000)")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--blocksize", type=int, default=8000, help="Block size in frames (default 0.5s at 16kHz)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not os.path.isdir(args.model):
        print(f"❌ Model path not found: {args.model}")
        sys.exit(1)

    transcripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "transcripts"))
    os.makedirs(transcripts_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(transcripts_dir, f"transcript_{timestamp}.txt")

    model = Model(args.model)
    rec = KaldiRecognizer(model, args.samplerate)
    rec.SetWords(True)

    print(f"🔹 Using model: {args.model}")
    print("🎤 Press Ctrl+C to stop. Listening to microphone...")

    try:
        with open(output_path, "a", encoding="utf-8") as fout:
            with sd.RawInputStream(samplerate=args.samplerate, blocksize=args.blocksize, dtype='int16',
                                   channels=1, callback=audio_callback, device=args.device):
                print(f"Listening (sample rate: {args.samplerate}) ...")
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text = res.get("text", "").strip()
                        if text:
                            print("\n✅ FINAL:", text)
                            fout.write(text + "\n")
                            fout.flush()
                    else:
                        pres = json.loads(rec.PartialResult())
                        partial = pres.get("partial", "").strip()
                        if partial:
                            print("⏳ PARTIAL: " + partial, end="\r")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Exiting...")
        final = json.loads(rec.FinalResult())
        final_text = final.get("text", "").strip()
        if final_text:
            print("✅ FINAL (on exit):", final_text)
            with open(output_path, "a", encoding="utf-8") as fout:
                fout.write(final_text + "\n")
        print(f"💾 Transcript saved to: {output_path}")
    except Exception as e:
        print("❌ Error:", str(e))
        raise

if __name__ == "__main__":
    main()
