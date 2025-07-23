import os
import faiss
import ollama
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Ordner mit .txt-Dokumenten
DOC_DIR = Path("ausgabe")  # z. B. Ausgabeordner deiner Text-Extraktion
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:12b"

# Lade Embedding-Modell
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# 1. 🔪 Dokumente in Chunks aufteilen
def load_chunks(doc_dir: Path, chunk_size=500):
    chunks = []
    sources = []
    for file in doc_dir.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
                sources.append(file.name)
    return chunks, sources

# 2. 🧠 Embedding + FAISS-Vektorspeicherung
def build_vector_index(chunks: List[str]):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# 3. 🔍 Semantische Suche
def retrieve(query: str, index, chunks, top_k=5):
    q_embed = embedder.encode([query])
    D, I = index.search(np.array(q_embed), top_k)
    return [chunks[i] for i in I[0]]

# 4. 🧠 Prompt an Ollama mit Kontext
def query_llm(query: str, context_chunks: List[str]) -> str:
    context = "\n---\n".join(context_chunks)
    prompt = f"""
Du bist ein hilfreiches Assistenzsystem. Verwende ausschließlich die folgenden Informationen zur Beantwortung der Nutzerfrage:

{context}

Frage: {query}
Antwort:"""
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# ▶️ RAG-Workflow
def run_rag(query: str):
    print("🔄 Lade Dokumente...")
    chunks, _ = load_chunks(DOC_DIR)
    print(f"📚 {len(chunks)} Text-Chunks geladen.")
    index, _ = build_vector_index(chunks)
    print("🔍 Suche relevante Informationen...")
    context = retrieve(query, index, chunks)
    print("🧠 Frage an LLM...")
    answer = query_llm(query, context)
    print("✅ Antwort:\n")
    print(answer)

# Beispielausführung
if __name__ == "__main__":
    frage = input("❓ Deine Frage: ")
    run_rag(frage)
