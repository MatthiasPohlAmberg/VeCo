import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] Dauer: {end - start:.2f} Sekunden")
        return result
    return wrapper


import subprocess

@timeit
def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

@timeit
def summarize_text_with_ollama(text, model="gemma3:12b"):
    # Modell muss vorher mit `ollama pull gemma:12b` geladen worden sein
    prompt = f"Fasse den folgenden Text zusammen:\n\n{text}"

    result = subprocess.run(
        ['ollama', 'run', model],
        input=prompt.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        print("Fehler:", result.stderr.decode())
        return None

    return result.stdout.decode()

if __name__ == "__main__":
    text = load_text_file("Tutorial_Leitungstrommel_LBL_full.txt")
    summary = summarize_text_with_ollama(text)
    print("\n--- Zusammenfassung ---\n", summary)
