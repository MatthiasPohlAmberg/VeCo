# -*- coding: utf-8 -*-
import os
import faiss
import ollama
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Directory with .txt documents (e.g., output folder from a text extraction step)
DOC_DIR = Path("output")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:12b"

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# 1. Split documents into chunks

def load_chunks(doc_dir: Path, chunk_size: int = 500) -> tuple[list[str], list[str]]:
    chunks: list[str] = []
    sources: list[str] = []
    for file in doc_dir.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
                sources.append(file.name)
    return chunks, sources

# 2. Create embeddings and store them in a FAISS index

def build_vector_index(chunks: List[str]):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# 3. Semantic search

def retrieve(query: str, index, chunks, top_k: int = 5) -> list[str]:
    q_embed = embedder.encode([query])
    distances, indices = index.search(np.array(q_embed), top_k)
    return [chunks[i] for i in indices[0]]

# 4. Send a prompt with context to Ollama

def query_llm(query: str, context_chunks: List[str]) -> str:
    context = "\n---\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant. Use only the following information to answer the user's question:

{context}

Question: {query}
Answer:"""
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# RAG workflow

def run_rag(query: str) -> None:
    print("Loading documents...")
    chunks, _ = load_chunks(DOC_DIR)
    print(f"Loaded {len(chunks)} text chunks.")
    index, _ = build_vector_index(chunks)
    print("Searching relevant information...")
    context = retrieve(query, index, chunks)
    print("Querying the LLM...")
    answer = query_llm(query, context)
    print("Answer:\n")
    print(answer)

# Example run

if __name__ == "__main__":
    question = input("Your question: ")
    run_rag(question)
