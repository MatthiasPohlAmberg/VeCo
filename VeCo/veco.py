import torch
import time
import logging
import whisper
from faiss import IndexFlatL2, IndexIDMap
import pdfplumber
import ollama
from pathlib import Path
import sys
import threading
import json
import pickle
import numpy as np
import pytesseract
from PIL import Image
from docx import Document
from pptx import Presentation
from moviepy import VideoFileClip
import tempfile
import os
from sentence_transformers import SentenceTransformer

# Logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Spinner:
    spinner_cycle = ["|", "/", "-", "\\"]

    def __init__(self, message="Processing"):
        self.stop_running = False
        self.thread = threading.Thread(target=self.run_spinner)
        self.message = message

    def run_spinner(self):
        i = 0
        while not self.stop_running:
            sys.stdout.write(f"\r{self.message}... {self.spinner_cycle[i % len(self.spinner_cycle)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_running = True
        self.thread.join()
        sys.stdout.write(f"\r{self.message}... done!\n")
        sys.stdout.flush()

class Vectorize:
    # def __init__(self, default_model="gemma3:12b"):
    def __init__(self, default_model="gemma3:12b", preload_path=None):  # New YZ
        self.default_model = default_model
        self.outputdb = []
        self.chunks = []
        spinner = Spinner("Initializing models")
        spinner.start()
        try:
            start_time = time.time()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("base", device=device)
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embedding_dim = self.embedder.get_sentence_embedding_dimension()
            self.faiss_index = IndexIDMap(IndexFlatL2(embedding_dim))
            self.check_ollama_models()
            logger.info(f"Models initialized in {time.time() - start_time:.2f} seconds.")
        finally:
            spinner.stop()

        # New YZ
        if preload_path:
            self.load_database(preload_path)

    def check_ollama_models(self):
        try:
            models = ollama.list()["models"]
            if not models:
                logger.warning("No models found in Ollama. Compression will be disabled.")
                self.use_compression = False
            else:
                self.use_compression = True
                logger.info(f"Ollama models available: {[m['name'] for m in models]}")
        except Exception as e:
            logger.warning(f"Ollama model check failed: {e}. Compression will be disabled.")
            self.use_compression = False

    def detect_input_type(self, inputfile):
        ext = Path(inputfile).suffix.lower()
        if ext in [".txt", ".pdf", ".doc", ".docx"]:
            return "text" if ext == ".txt" else "pdf" if ext == ".pdf" else "word"
        elif ext == ".pptx":
            return "pptx"
        elif ext in [".jpg", ".jpeg", ".png"]:
            return "image"
        elif ext in [".mp3", ".wav"]:
            return "audio"
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            return "video"
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def vectorize(self, inputfile, **kwargs):
        use_compression = kwargs.get("use_compression", False)
        model = kwargs.get("model", self.default_model)
        spinner = Spinner("Vectorizing input")
        spinner.start()
        try:
            input_type = self.detect_input_type(inputfile)
            logger.info(f"Detected input type: {input_type}")
            selected_model = model if model else self.default_model

            if input_type in ["text", "pdf", "word", "pptx"]:
                raw_text = self.extract_text(inputfile, input_type)
            elif input_type == "image":
                raw_text = self.extract_text_from_image(inputfile)
            elif input_type == "audio":
                raw_text = self.extract_text_from_audio(inputfile)
            elif input_type == "video":
                raw_text = self.extract_text_from_video(inputfile)
            else:
                raw_text = ""

            if use_compression:
                compressed_text = self.ask_llm(self.build_compression_prompt(raw_text), selected_model)
                text_for_output = compressed_text
            else:
                text_for_output = raw_text

            vector = self.vectorize_text(text_for_output)

            self.outputdb.append({
                "id": len(self.outputdb),
                "vector": vector.tolist(),
                "text": text_for_output,
                "source": str(inputfile)
            })
        finally:
            spinner.stop()

    def extract_text(self, inputfile, input_type):
        if input_type == "text":
            return Path(inputfile).read_text(encoding="utf-8")
        elif input_type == "pdf":
            text = ""
            with pdfplumber.open(inputfile) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        elif input_type == "word":
            doc = Document(inputfile)
            return "\n".join([para.text for para in doc.paragraphs])
        elif input_type == "pptx":
            presentation = Presentation(inputfile)
            text = ""
            for slide in presentation.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        else:
            return ""

    def extract_text_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image, lang="deu")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def extract_text_from_audio(self, audio_path):
        result = self.whisper_model.transcribe(audio_path, language="en")
        return result["text"]

    def extract_text_from_video(self, video_path):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(tmp_audio.name, logger=None)
                text = self.extract_text_from_audio(tmp_audio.name)
            os.remove(tmp_audio.name)
            return text
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return ""

    def vectorize_text(self, text):
        vector = self.embedder.encode([text], convert_to_numpy=True)
        ids = np.arange(vector.shape[0])
        self.faiss_index.add_with_ids(vector, ids)
        return vector

    def build_compression_prompt(self, raw_text):
        return f"Please compress the following text:\n\n{raw_text}"

    def ask_llm(self, prompt, model):
        messages = [{"role": "user", "content": prompt}]
        try:
            response = ollama.chat(model=model, messages=messages)
            content = response.get("message", {}).get("content", "")
            if not content:
                logger.warning("LLM returned empty response.")
            return content
        except Exception as e:
            logger.error(f"LLM interaction failed: {e}")
            return ""

    def retrieve_context(self, query: str, top_k=5):
        q_embed = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.faiss_index.search(q_embed, top_k)
        context_passages = []
        for idx in I[0]:
            if 0 <= idx < len(self.outputdb):
                context_passages.append(self.outputdb[idx]["text"])
        return "\n\n".join(context_passages)

    def query_with_context(self, question: str, model=None, top_k=5):
        selected_model = model if model else self.default_model
        context = self.retrieve_context(question, top_k)

        prompt = f"""Use the following context to answer the question.
            Context:
            {context}
            Question:
            {question}
            Answer:"""

        spinner = Spinner("Asking LLM via Ollama")
        spinner.start()
        try:
            response = self.ask_llm(prompt, selected_model)
            return response
        finally:
            spinner.stop()
    
    def save_database(self, filepath, format="json"):
        # Load existing data if the file exists
        if Path(filepath).exists():
            self.load_database(filepath, format)

        # Now, save the updated database
        data = {
            "outputdb": self.outputdb
        }
        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")


    def load_database(self, filepath, format="json"):
        if format == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif format == "pickle":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # New YZ
        # self.outputdb = data.get("outputdb", [])
        # for entry in self.outputdb:
        #     vector = np.array(entry["vector"])
        #     self.faiss_index.add_with_ids(vector.reshape(1, -1), np.array([entry["id"]]))

        existing_ids = {entry["id"] for entry in self.outputdb}

        for entry in data.get("outputdb", []):
            if entry["id"] not in existing_ids:
                vector = np.array(entry["vector"])
                self.faiss_index.add_with_ids(vector.reshape(1, -1), np.array([entry["id"]]))
                self.outputdb.append(entry)


if __name__ == "__main__":
    # veco = Vectorize()
    veco = Vectorize(preload_path="vector_db.json")  # New YZ
    input_file = "Vectorizing_the_Company.pdf"  # Change this to your test file
    veco.vectorize(input_file, use_compression=False)
    veco.save_database("vector_db.json", format="json")

    # Example query
    question = "What is the company's strategy?"
    answer = veco.query_with_context(question)
    print("\nAnswer:\n", answer)
