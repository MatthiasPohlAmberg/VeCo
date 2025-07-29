import torch
import time
import logging
import whisper
from faiss import IndexFlatL2, IndexIDMap
import pdfplumber
import base64
import ollama
from pathlib import Path
import sys
import threading
from sentence_transformers import SentenceTransformer
import numpy as np

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
        sys.stdout.write(f"\r{self.message}... done!")
        sys.stdout.flush()

class Vectorize:
    def __init__(self):
        spinner = Spinner("Initializing models")
        spinner.start()
        try:
            # Initialize models
            start_time = time.time()
            # Check if CUDA is available
            if torch.cuda.is_available():
                logger.info(f"CUDA is available")
                logger.info(f"Device count: {torch.cuda.device_count()}")
                logger.info(f"Current device: {torch.cuda.current_device()}")
                logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            else:
                logger.info("CUDA is NOT available :(")
            
            # Patch for torch.load to enforce weights_only=False (if needed)
            orig_load = torch.load
            def patch_load(*args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return orig_load(*args, **kwargs)
            torch.load = patch_load
            # Select device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("turbo", device=device)
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embedding_dim = self.embedder.get_sentence_embedding_dimension()
            self.faiss_index = IndexIDMap(IndexFlatL2(embedding_dim))  # Use IndexIDMap for dynamic IDs
            end_time = time.time()
            logger.info(f"Models initialized in {end_time - start_time:.2f} seconds.")
        finally:
            spinner.stop()

    def vectorize(self, outputdb, inputfile, use_compression=False):
        """
        Automatically detect input type and process it.
        :param outputdb: Database for output vectors
        :param inputfile: Input file
        :param use_compression: Whether to use an LLM for compression
        """
        spinner = Spinner("Vectorizing input")
        spinner.start()
        try:
            start_time = time.time()
            try:
                input_type = self.detect_input_type(inputfile)
                logger.info(f"Input type detected: {input_type}")

                if input_type in ["text", "pdf", "word"]:
                    raw_text = self.extract_text(inputfile, input_type)
                    logger.info(f"Extracted text: {raw_text}")  # Log the extracted text
                    if use_compression:
                        compressed_text = self.ask_gemma(self.build_compression_prompt(raw_text))
                        vector = self.vectorize_text(compressed_text)
                    else:
                        vector = self.vectorize_text(raw_text)
                    outputdb.append(vector)
                elif input_type == "image":
                    if use_compression:
                        prompt = self.build_image_interpretation_prompt(inputfile)
                        interpreted_data = self.ask_gemma(prompt)
                        vector = self.vectorize_text(interpreted_data)
                    else:
                        raw_text = self.extract_text_from_image(inputfile)
                        logger.info(f"Extracted text: {raw_text}")  # Log the extracted text
                        vector = self.vectorize_text(raw_text)
                    outputdb.append(vector)
                elif input_type == "audio":
                    raw_text = self.extract_text_from_audio(inputfile)
                    logger.info(f"Extracted text: {raw_text}")
                # ToDo type Video
                # ToDo type Excel
                else:
                    logger.warning(f"Unsupported input type: {input_type}. Routine läuft weiter.")
            except ValueError as e:
                logger.warning(f"Error: {e}. Next file or end.")
            end_time = time.time()
            logger.info(f"Processing completed in {end_time - start_time:.2f} seconds.")
        finally:
            spinner.stop()

    def detect_input_type(self, inputfile):
        """
        Detect the input type based on file extension.
        """
        spinner = Spinner("Detecting input type")
        spinner.start()
        try:
            start_time = time.time()
            ext = Path(inputfile).suffix.lower()
            if ext in [".txt", ".pdf", ".doc", ".docx"]:
                input_type = "text" if ext == ".txt" else "pdf" if ext == ".pdf" else "word"
            elif ext in [".jpg", ".jpeg", ".png"]:
                input_type = "image"
            elif ext in [".mp3", ".wav"]:
                input_type = "audio"
            elif ext in [".mp4", ".avi"]:
                input_type = "video"
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            end_time = time.time()
            logger.info(f"Input type detection completed in {end_time - start_time:.2f} seconds.")
            return input_type
        finally:
            spinner.stop()

    def extract_text(self, inputfile, input_type):
        """
        Extract text based on the input type.
        """
        spinner = Spinner("Extracting text")
        spinner.start()
        try:
            start_time = time.time()
            if input_type == "text":
                text = Path(inputfile).read_text(encoding="utf-8")
            elif input_type == "pdf":
                text = self.extract_text_from_pdf(inputfile)
            elif input_type == "word":
                text = self.extract_text_from_word(inputfile)
            else:
                raise ValueError(f"Unsupported input type for text extraction: {input_type}")
            end_time = time.time()
            logger.info(f"Text extraction completed in {end_time - start_time:.2f} seconds.")
            return text
        finally:
            spinner.stop()

    def extract_text_from_pdf(self, filepath):
        """Extract text from PDF files."""
        spinner = Spinner("Extracting text from PDF")
        spinner.start()
        try:
            start_time = time.time()
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + ""
            end_time = time.time()
            logger.info(f"PDF text extraction completed in {end_time - start_time:.2f} seconds.")
            return text
        finally:
            spinner.stop()

    def extract_text_from_word(self, filepath):
        """Dummy function for extracting text from Word documents."""
        spinner = Spinner("Extracting text from Word document")
        spinner.start()
        try:
            start_time = time.time()
            text = "Word document content"  # Implementation required
            end_time = time.time()
            logger.info(f"Word text extraction completed in {end_time - start_time:.2f} seconds.")
            return text
        finally:
            spinner.stop()

    def extract_text_from_image(self, image_path):
        """Dummy function for extracting text from images."""
        spinner = Spinner("Extracting text from image")
        spinner.start()
        try:
            start_time = time.time()
            text = "OCR text from image"  # Implementation required
            end_time = time.time()
            logger.info(f"Image text extraction completed in {end_time - start_time:.2f} seconds.")
            return text
        finally:
            spinner.stop()

    def build_compression_prompt(self, raw_text):
        """
        Create a prompt for compression using an LLM.
        """
        spinner = Spinner("Building compression prompt")
        spinner.start()
        try:
            start_time = time.time()
            prompt = f"Please compress the following text: {raw_text} "
            end_time = time.time()
            logger.info(f"Compression prompt created in {end_time - start_time:.2f} seconds.")
            return prompt
        finally:
            spinner.stop()

    def extract_text_from_audio(self, audio_path):
        """
        Extract text from audio files using Whisper.
        """
        spinner = Spinner("Extracting text from audio")
        spinner.start()
        try:
            start_time = time.time()
            result = self.whisper_model.transcribe(audio_path, language="de")
            text = result["text"]
            end_time = time.time()
            logger.info(f"Audio transcription completed in {end_time - start_time:.2f} seconds.")
            return text
        finally:
            spinner.stop()
    
    def build_image_interpretation_prompt(self, image_path):
        """
        Create a prompt for image interpretation using an LLM.
        """
        spinner = Spinner("Building image interpretation prompt")
        spinner.start()
        try:
            start_time = time.time()
            encoded_image = self.encode_image_base64(image_path)
            prompt = f"Please interpret the content of this image:{encoded_image}"
            end_time = time.time()
            logger.info(f"Image interpretation prompt created in {end_time - start_time:.2f} seconds.")
            return prompt
        finally:
            spinner.stop()

    def ask_gemma(self, prompt):
        """
        Communicate with the LLM model.
        """
        spinner = Spinner("Communicating with LLM")
        spinner.start()
        try:
            start_time = time.time()
            messages = [{"role": "user", "content": prompt}]
            response = ollama.chat(model="gemma3:12b", messages=messages)
            end_time = time.time()
            logger.info(f"LLM communication completed in {end_time - start_time:.2f} seconds.")
            return response["message"]["content"]
        finally:
            spinner.stop()

    def encode_image_base64(self, image_path):
        """
        Encode an image in Base64.
        """
        spinner = Spinner("Encoding image to Base64")
        spinner.start()
        try:
            start_time = time.time()
            with open(image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
            end_time = time.time()
            logger.info(f"Image encoded in Base64 in {end_time - start_time:.2f} seconds.")
            return encoded_image
        finally:
            spinner.stop()

    def vectorize_text(self, text):
        """Vectorize text."""
        spinner = Spinner("Vectorizing text")
        spinner.start()
        try:
            start_time = time.time()
            vector = self.embedder.encode([text], convert_to_numpy=True)
            ids = np.arange(vector.shape[0])
            self.faiss_index.add_with_ids(vector, ids)
            end_time = time.time()
            logger.info(f"Text vectorization completed in {end_time - start_time:.2f} seconds.")
            return vector
        finally:
            spinner.stop()

    def load_chunks(self, doc_dir: Path, chunk_size=500):
        spinner = Spinner("Loading chunks")
        spinner.start()
        try:
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
        finally:
            spinner.stop()

    def build_vector_index(self, chunks):
        spinner = Spinner("Building vector index")
        spinner.start()
        try:
            embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
            ids = np.arange(embeddings.shape[0])
            self.faiss_index.add_with_ids(embeddings, ids)
            return self.faiss_index, embeddings
        finally:
            spinner.stop()

    def retrieve(self, query: str, top_k=5):
        spinner = Spinner("Retrieving results")
        spinner.start()
        try:
            q_embed = self.embedder.encode([query])
            D, I = self.faiss_index.search(np.array(q_embed), top_k)
            return [self.chunks[i] for i in I[0]]
        finally:
            spinner.stop()

# Example usage
if __name__ == "__main__":
    veco = Vectorize()
    outputdb = []  # Example database
    veco.vectorize(outputdb, "Vectorizing_the_Company.pdf", use_compression=False)
    print(outputdb)
    """
    # Example for FAISS retrieval
    DOC_DIR = Path("ausgabe")
    chunks, _ = veco.load_chunks(DOC_DIR)
    veco.build_vector_index(chunks)
    query = "Deine Frage hier"
    results = veco.retrieve(query)
    print(results)"""