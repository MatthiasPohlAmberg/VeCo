import whisper
import os
import torch
import time

# Prüfen, ob CUDA verfügbar ist
if torch.cuda.is_available():
    print("CUDA is available! :)")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available :(")

# Patch für torch.load, um weights_only=False zu erzwingen (sofern benötigt)
orig_load = torch.load
def patch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return orig_load(*args, **kwargs)
torch.load = patch_load

# Verzeichnis mit Audiodateien (MP3, WAV, ...)
folder_path = r"C:\Users\MatthiasPohl\Desktop\KI\whisper\Process"

# Device auswählen
device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper-Modell laden (z. B. "turbo")
print("Lade Whisper-Modell ...")
model_load_start = time.time()
model = whisper.load_model("turbo", device=device)
model_load_end = time.time()
print(f"Modell geladen in {model_load_end - model_load_start:.2f} Sekunden auf {device}.")

# Alle Dateien im Ordner durchgehen
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".mp3", ".wav", ".mp4")):
        audio_path = os.path.join(folder_path, filename)
        base_name, _ = os.path.splitext(filename)
        
        # Einen kombinierten Text-Dateinamen erstellen (z. B. "audiofile_full.txt")
        output_txt_path = os.path.join(folder_path, f"{base_name}_full.txt")
        
        print(f"\nVerarbeite: {audio_path}")
        start_time = time.time()
        
        # Transkription starten (Whisper teilt das Audio intern in Chunks auf)
        result = model.transcribe(audio_path)
        
        # Abruf der automatisch erstellten Segmente
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
        print(f"Transkription abgeschlossen in {elapsed:.2f} Sekunden.")
        print(f"Ausgabe gespeichert in: {output_txt_path}")
