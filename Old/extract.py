import base64
import time
from pathlib import Path
import pdfplumber
import ollama

# Directories
INPUT_DIR = Path(r"C:\Users\MatthiasPohl\Desktop\KI\data_extract\Eingabe")
OUTPUT_DIR = Path(r"C:\Users\MatthiasPohl\Desktop\KI\data_extract\Ausgabe")
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
SUPPORTED_TEXT_EXTS = [".pdf", ".txt"]

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def ask_gemma(content, images=None):
    messages = [{"role": "user", "content": content}]
    if images:
        messages[0]["images"] = images
    response = ollama.chat(model="gemma3:12b", messages=messages)
    return response['message']['content']

def build_prompt(doc_text=None):
    if doc_text:
        return (
            "Please extract the content from the following document text. "
            "Document content:\n"""
" + doc_text[:8000] + "
""""
        )
    else:
        return (
            "Please extract the full textual content from this image. "
            "Ignore artefacts"
        )

def process_file(filepath: Path):
    ext = filepath.suffix.lower()
    output_path = OUTPUT_DIR / (filepath.stem + ".txt")

    print(f">> Processing: {filepath.name}")
    t0 = time.perf_counter()

    try:
        # Step 1: OCR / text preparation
        t1 = time.perf_counter()
        if ext in SUPPORTED_IMAGE_EXTS:
            encoded_image = encode_image_base64(filepath)
            prompt = build_prompt()
        elif ext == ".pdf":
            raw_text = extract_text_from_pdf(filepath)
            prompt = build_prompt(raw_text)
            encoded_image = None
        elif ext == ".txt":
            raw_text = filepath.read_text(encoding="utf-8")
            prompt = build_prompt(raw_text)
            encoded_image = None
        else:
            print(f"Unsupported file type: {filepath.name}")
            return
        t2 = time.perf_counter()
        
        # Step 2: Call the LLM
        result = ask_gemma(prompt, images=[encoded_image] if encoded_image else None)
        t3 = time.perf_counter()

        # Step 3: Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        t4 = time.perf_counter()

        print(f"OCR/extraction: {t2 - t1:.2f}s")
        print(f" LLM (gemma): {t3 - t2:.2f}s")
        print(f" Writing: {t4 - t3:.2f}s")
        print(f"Saved: {output_path.name}")

    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")

    finally:
        print(f"Total: {time.perf_counter() - t0:.2f}s\n")

def main():
    overall_start = time.perf_counter()
    for file in INPUT_DIR.iterdir():
        if file.is_file():
            process_file(file)
    print(f"Total runtime: {time.perf_counter() - overall_start:.2f} seconds")

if __name__ == "__main__":
    main()
