# VeCo# - Vectorizing the company

This Python module allows you to vectorize text and media files using machine learning models. The main functionality includes extracting text from different file types (PDF, DOCX, PPTX, images, audio, video), vectorizing that text, and storing the results in a database for further querying.

## Features
- Extracts text from various file types, including PDFs, DOCX, PPTX, images, audio, and video.
- Supports vectorization using SentenceTransformer.
- Optionally compresses text using LLMs via Ollama.
- Saves vectorized data to JSON or Pickle format for persistence.
- Provides context-based querying via LLM.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MatthiasPohlAmberg/VeCo.git
2. Install dependencies:
cd VeCo
pip install -r requirements.txt

3. Usage

from VeCo.vectorize import Vectorize

# Initialize Vectorizer
vec = Vectorize()

# Vectorize a file
vec.vectorize("your_file.pdf")

# Query the database
response = vec.query_with_context("What is the company's strategy?")
print(response)

# Save the database
vec.save_database("vector_db.json", format="json")