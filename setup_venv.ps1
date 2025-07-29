# Prüft, ob Python 3.10 verfügbar ist
python --version
# Optional: prüfe ob der Pfad zu Python 3.10 korrekt ist
$pythonPath = (Get-Command python).Source
Write-Host "Python-Path: $pythonPath"

# Erstellt virtuelle Umgebung mit Python 3.10
py -3.10 -m venv .venv

# Aktiviert die Umgebung
. .venv\Scripts\Activate.ps1

# Installiert Pakete

pip install -r requirements.txt

Write-Host "Virtual environment with Python 3.10 created!"
