# Meeting Summarizer 📝  

This repository contains scripts and utilities for **speech recognition, diarization, and summarization of meetings**.  

## 📂 Files in this repo  

- `record_test.py` → Record audio for testing  
- `whisper_evaluate.py` → Evaluate Whisper model  
- `whisper_vad_realtime.py` → Real-time speech detection with Whisper  
- `summarizer.py` → Summarizes transcriptions  
- `realtime_vosk.py` → Real-time transcription using Vosk  
- `WER_calculator.py` → Calculate Word Error Rate  
- `librispeech_to_csv.py` → Convert LibriSpeech dataset to CSV  
- `generate_dataset_csv.py` → Generate dataset CSV files  
- `evaluate_whisper.py` → Evaluate Whisper model accuracy  
- `evaluate_vosk.py` → Evaluate Vosk model accuracy  
- `batch_evaluate_whisper.py` → Batch evaluation with Whisper  
- `batch_evaluate_vosk.py` → Batch evaluation with Vosk  
- `diarize_whisper.py` → Speaker diarization with Whisper  
- `flac_to_wav.py` → Convert FLAC audio to WAV  
- `calculate_der.py` → Calculate Diarization Error Rate (DER)  
- `average_wer.py` → Calculate average WER across files  

## 🚀 How to Use  
1. Clone the repo:  
   ```bash
   git clone https://github.com/sairishikamora/meeting_summarizer.git
   cd meeting_summarizer
2.install dependencies:
pip install -r requirements.txt
3.Run a script:
python whisper_evaluate.py

## 👩‍💻 Author
- **Rishika Mora**  
B.Tech(Information Technology), SNIST  
[GitHub Profile](https://github.com/sairishikamora)

## 📂 Repository Features
This repository contains multiple scripts for:
- Meeting summarization
- Transcription (Whisper & Vosk)
- Evaluation (WER, DER)
- Dataset preparation
- 
## 📜 License
This project is open-source and available under the MIT License.
