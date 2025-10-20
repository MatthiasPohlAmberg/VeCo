import whisper
import os
import torch
import time

# Check whether CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! :)")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available :(")

# Patch torch.load to enforce weights_only=False (if needed)
orig_load = torch.load
def patch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return orig_load(*args, **kwargs)
torch.load = patch_load

# Directory containing audio files (MP3, WAV, ...)
folder_path = r"C:\Users\MatthiasPohl\Desktop\KI\whisper\Process"

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model (e.g., "turbo")
print("Loading Whisper model ...")
model_load_start = time.time()
model = whisper.load_model("turbo", device=device)
model_load_end = time.time()
print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds on {device}.")

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".mp3", ".wav", ".mp4")):
        audio_path = os.path.join(folder_path, filename)
        base_name, _ = os.path.splitext(filename)
        
        # Create a combined text filename (e.g., "audiofile_full.txt")
        output_txt_path = os.path.join(folder_path, f"{base_name}_full.txt")
        
        print(f"\nProcessing: {audio_path}")
        start_time = time.time()
        
        # Start transcription (Whisper handles chunking internally)
        result = model.transcribe(audio_path)
        
        # Retrieve automatically generated segments
        segments = result.get("segments", [])
        
        # Alle Segmente (mit Zeitstempeln) in eine kombinerte Datei schreiben
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                start = seg["start"]
                end = seg["end"]
                text = seg["text"].strip()
                f.write(f"[{start:.2f} - {end:.2f}] {text}\n")
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Transcription finished in {elapsed:.2f} seconds.")
        print(f"Output saved to: {output_txt_path}")
