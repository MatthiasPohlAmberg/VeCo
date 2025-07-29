#!/bin/bash

# Prüft Python-Version
python3.10 --version || { echo "❌ Python 3.10 not found"; exit 1; }

# Erstellt virtuelle Umgebung
python3.10 -m venv .venv

# Aktiviert venv
source .venv/bin/activate

# Installiert Pakete
pip install -r requirements.txt

echo "Virtual environment with Python 3.10 created!"
