# VeCo

**VeCo** ist ein Python-Toolkit (Python 3.10–3.11), um unterschiedlichste Dokumente – Text, PDF, Word, PowerPoint, Bilder, Audio und Video – in Vektor­repräsentationen zu transformieren.  
Die Embeddings werden in einem FAISS-Index gespeichert und können optional zusätzlich in JSON (Fallback), SQLite oder MongoDB persistiert werden.  
Mit der integrierten RAG-Schnittstelle lassen sich Wissensdatenbanken direkt über **Ollama-LLMs** abfragen.

## Features

- Automatische **Input-Erkennung**: Text, PDF, Word, PowerPoint, Bild, Audio, Video
- **Text-Extraktion**: `pdfplumber`, `python-docx`, `python-pptx`, `pytesseract`, `moviepy`, `whisper`
- **Speaker Diarization** (optional über `veco_diarization.py` Integration)
- **Vision-Erweiterungen**:
  - OCR via `pytesseract`
  - CNN-Klassifikation (torchvision ResNet)
  - Externe Bildbeschreibung (optional via `veco_pic_describe`)
- **Chunking mit Overlap** für RAG-fähige Embeddings
- **Optionale Summaries** über Ollama-LLMs (werden separat gespeichert, nicht als Embedding-Basis)
- **FAISS-Index** für effizientes Retrieval
- **Persistenz**: JSON (Fallback, Stand-alone), SQLite oder MongoDB
- **RAG-Queries**: End-to-End Abfrage (`query()`) → Kontext holen + Ollama-Antwort erzeugen

## Project Structure

```
.
├── VeCo/
│   ├── __init__.py
│   ├── veco.py            # Core vectorization library
│   ├── storages.py        # Optionale SQLite/MongoDB Backends
│   ├── veco_diarization.py# Optionale Speaker Diarization Pipeline
│   └── veco_pic_describe/ # Optionales Projekt für Bildbeschreibung
├── tests/
│   └── veco_test.py       # Example usage script
├── requirements.txt
├── pyproject.toml / setup.py
├── test_data/             # Sample files for testing
├── vector_db.json         # Beispiel-DB (JSON Fallback)
└── UML/                   # Architekturdiagramme
```

## Dependencies

- Python 3.10 oder 3.11
- `torch`, `torchaudio`, `torchvision` (CPU-Wheels via PyPI; für CUDA bitte offizielle PyTorch-Anleitung nutzen)
- `sentence-transformers`
- `faiss-cpu`
- `openai-whisper`
- `pdfplumber`
- `pytesseract`
- `pillow`
- `moviepy`
- `python-docx`
- `python-pptx`
- `numpy` & `scipy`
- `ollama`
- `webrtcvad-wheels`, `librosa`, `soundfile`, `speechbrain`
- (Siehe `requirements.txt` / `pyproject.toml` für exakte Versionen)

## Installation

### 1. Virtuelle Umgebung anlegen

```bash
python3.10 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 2. Basis-Dependencies installieren

```bash
pip install -r requirements.txt
```

oder mit `pyproject.toml`:

```bash
pip install .
```

### 3. PyTorch-Konfiguration (optional)

Folge der offiziellen [PyTorch Installationsanleitung](https://pytorch.org/get-started/locally/) für deine GPU/CPU-Konfiguration. Für eine reine CPU-Installation genügt das Standard-`pip install` der Requirements.

## Usage

### Beispielskript (`tests/veco_test.py`)

```bash
python tests/veco_test.py
```

Dieses Skript lädt oder erstellt `vector_db.json`, vektorisiert alle Dateien im Ordner `test_data/` und speichert die aktualisierte Datenbank.

### Direkte Nutzung in Python

```python
from VeCo import Vectorize

# JSON-Fallback
veco = Vectorize(preload_json_path="vector_db.json")

# Datei vektorisieren
veco.vectorize("path/to/file.pdf", use_compression=True)

# Datenbank speichern
veco.save_database("vector_db.json")

# RAG-Query (mit Ollama)
res = veco.query(
    database="vector_db.json",
    frage="Worum geht es im Dokument?",
    llm_model="gemma3:12b",
)
print(res["answer"])
```

## Architecture

Die zentrale Klasse ist `Vectorize`:

- **Input detection**: erkennt Dateityp
- **Text extraction**: nutzt spezifische Libraries pro Typ
- **Optional compression**: Summaries via Ollama
- **Chunking**: Texte werden in kleine Overlap-Chunks zerlegt
- **Embedding**: `sentence-transformers`
- **Storage**: FAISS-Index + JSON/SQLite/Mongo
- **Retrieval**: `retrieve_context()` liefert Chunks
- **Query**: `query()` kombiniert Kontext + Ollama-Antwort → echtes RAG

Erweiterungen wie Speaker Diarization (`veco_diarization.py`) oder Bildbeschreibung (`veco_pic_describe`) sind modular integriert.
