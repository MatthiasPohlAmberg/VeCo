# veco-ai

veco-ai is a Python toolkit (Python 3.10-3.11) that converts a broad range of document types - text, PDF, Word, PowerPoint, images, audio, and video - into vector representations that can be queried through Retrieval Augmented Generation (RAG).  
Embeddings are stored inside a FAISS index and can optionally be persisted to JSON (fallback), SQLite, or MongoDB. The integrated RAG interface lets you query knowledge bases via local **Ollama** models.

## Features

- Automatic **input detection** for text, PDF, Word, PowerPoint, images, audio, and video
- **Text extraction** via `pdfplumber`, `python-docx`, `python-pptx`, `pytesseract`, `moviepy`, and `whisper`
- **Speaker diarization** (optional through `veco_diarization.py`)
- **Vision extensions**:
  - OCR via `pytesseract`
  - CNN classification (torchvision ResNet)
  - External image captioning (optional via `veco_pic_describe`)
- **Chunking with overlap** for RAG-ready embeddings
- **Optional summaries** generated with Ollama models (stored separately, never used as embedding input)
- **FAISS index** for efficient retrieval
- **Persistence** backends: JSON (fallback, stand-alone), SQLite, or MongoDB
- **RAG queries**: End-to-end helper (`query()`) that retrieves context and triggers an Ollama response

## Project Structure

```
.
|-- veco_ai/
|   |-- __init__.py
|   |-- veco_ai.py              # Core vectorization library
|   |-- veco_diarization.py  # Optional speaker diarization pipeline
|   `-- veco_pic_describe.py # Optional image captioning helpers
|-- test/
|   `-- veco_test.py         # Example usage script
|-- requirements.txt
|-- pyproject.toml / setup.py
|-- test_data/               # Sample files for testing
|-- vector_db.json           # Example JSON database (fallback storage)
`-- UML/                     # Architecture diagrams
```

## Dependencies

- Python 3.10 or 3.11
- `torch`, `torchaudio`, `torchvision` (CPU wheels via PyPI; follow the official PyTorch guide for CUDA)
- `sentence-transformers`
- `faiss-cpu`
- `openai-whisper`
- `pdfplumber`
- `pytesseract`
- `pillow`
- `moviepy`
- `python-docx`
- `python-pptx`
- `numpy` and `scipy`
- `ollama`
- `webrtcvad-wheels`, `librosa`, `soundfile`, `speechbrain`
- (See `requirements.txt` / `pyproject.toml` for exact versions)

## Installation

### 1. Create a virtual environment

```bash
./setup_venv.ps1
# or
python3.11 -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS
```

### 2. Install the base dependencies

```bash
pip install veco_ai
```

For local development instead of the published wheel, install from source:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### 3. Configure PyTorch (optional)

Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your GPU/CPU setup.  
For a CPU-only environment the default `pip install` from the requirements is sufficient.

## Usage

### Example script (`tests/veco_test.py`)

```bash
python tests/veco_test.py
```

The script loads or creates `vector_db.json`, vectorizes all files in the `test_data/` folder, and saves the updated database.

### Direct usage in Python

```python
from veco_ai import Vectorize

# JSON fallback backend
veco = Vectorize(preload_json_path="vector_db.json")

# Vectorize a file
veco.vectorize("path/to/file.pdf", use_compression=True)

# Persist the database
veco.save_database("vector_db.json")

# Run a RAG query (Ollama required)
res = veco.query(
    database="vector_db.json",
    question="What is this document about?",
    llm_model="gemma3:12b",
)
print(res["answer"])
```

## Architecture

The central class is `Vectorize`:

- **Input detection**: identifies the file type
- **Text extraction**: uses type-specific libraries
- **Optional compression**: generates summaries through Ollama
- **Chunking**: splits text into overlapping segments
- **Embedding**: performed with `sentence-transformers`
- **Storage**: FAISS index plus JSON/SQLite/MongoDB backends
- **RAG**: retrieves relevant context and optionally queries an Ollama model

## Development

Install the development extras to run linting and tests:

```bash
pip install .[dev]
pytest
```

## License

The project is released under the terms of [CC0 1.0 Universal](LICENSE).
