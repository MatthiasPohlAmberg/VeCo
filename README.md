# VeCo

VeCo is a Python 3.10 based toolkit for transforming various documents—text, PDFs, images, audio, and video—into vector representations. These embeddings can be stored in a FAISS index for later retrieval or analysis.

## Project Structure

```
.
├── veco.py           # Core vectorization library
├── veco_test.py      # Example usage script
├── requirements.txt  # Python dependencies
├── test_data/        # Sample files for testing
├── vector_db.json    # Example output vector database
├── setup_venv.sh/.bat/.ps1  # Environment setup scripts
├── run_veco_venv.*   # Scripts to activate the virtual environment
├── Evaluation/       # Evaluation resources
├── Old/              # Legacy scripts
└── UML/              # Architecture diagrams
```

## Dependencies

- Python 3.10
- torch
- sentence-transformers
- faiss-cpu
- openai-whisper
- pdfplumber
- pytesseract
- pillow
- moviepy
- python-docx
- python-pptx
- numpy
- ollama
- (See `requirements.txt` for exact versions)

## Usage

1. **Create and activate a virtual environment (Python 3.10).**

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```

   Alternatively, run one of the provided `setup_venv` scripts.

2. **Install dependencies.**

   ```bash
   pip install -r requirements.txt
   ```

3. **Vectorize files.**

   ```bash
   python veco_test.py
   ```

   This script loads or creates `vector_db.json`, vectorizes every file in `test_data/`, and saves the updated database.

   To use the library directly:

   ```python
   from veco import Vectorize

   veco = Vectorize()
   veco.vectorize("path/to/file.pdf", use_compression=False)
   veco.save_database("vector_db.json", format="json")
   ```

## Architecture

VeCo centers around the `Vectorize` class:

- **Input detection** determines whether the source is text, PDF, Word, PowerPoint, image, audio, or video.
- **Text extraction** uses libraries like `pdfplumber`, `python-docx`, `python-pptx`, `pytesseract`, and `moviepy` to obtain raw text from different formats.
- **Optional compression** leverages an Ollama-hosted language model to summarize large inputs.
- **Embedding** is handled by `sentence-transformers`, producing numerical vectors.
- **Storage and retrieval** of embeddings use a FAISS index, with serialized results saved to JSON or pickle.

This architecture makes VeCo adaptable to multiple data sources while providing a consistent vector representation for downstream applications.
