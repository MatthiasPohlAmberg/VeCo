
from veco import Vectorize
from pathlib import Path

def main():
    # 1) Projekt-Root & Pfade (alles relativ)
    current_directory = Path(__file__).parent
    project_root = Path(r'E:\01_Dokumente\12_Startup-Mentor\MP_Data') #current_directory.parent
    db_path = project_root / "vector_db.json"
    test_data_directory = project_root # / "test_data"

# 2. Load existing vector database (if available)
db_path = "vector_db.json"
try:
    veco.load_database(db_path, format="json")
    print("Existing database loaded.")
except FileNotFoundError:
    print("No existing database found. A new one will be created.")

    # 2) Vectorizer initialisieren (JSON-Fallback vorgeben)
    veco = Vectorize(preload_json_path=str(db_path))

    # 3) Vorhandene DB laden (wirft keine Exception, startet leer wenn nicht vorhanden)
    veco.load_database(str(db_path))
    print("Database loaded (or initialized empty).")

    # 4) Alle Dateien in test_data vektorisieren
    """
    if test_data_directory.is_dir():
        for file in sorted(test_data_directory.iterdir()):
            if file.is_file():
                try:
                    veco.vectorize(str(file), use_compression=False)
                    print(f"OK: {file.name} vectorized")
                except Exception as e:
                    print(f"ERROR: {file.name}: {e}")
    else:
        print(f"Missing test_data directory: {test_data_directory}")

    # 5) JSON speichern
    veco.save_database(str(db_path))
    print(f"Database updated and saved → {db_path}")
    """
    """
    # 6) Optional: Mini-Retrieval-Test (ohne LLM)
    try:
        res = veco.query_with_context("Worum geht es?", top_k=5, include_summary=True)
        ctx_sources = [c.get("source") for c in res.get("contexts", [])]
        print("Sample retrieval sources:", ctx_sources)
    except Exception as e:
        print(f"Retrieval error: {e}")
    """   
    # Res mit LLM (falls Ollama läuft)
    rag = veco.query(database=str(db_path), frage="Erstelle mir für einen Grafiker ein Portrait auf Basis des Unternehmens und der beantworteten Fragen. Fasse hierbei die Aussagen erst zusammen und erstelle auf basis dieser ein Profil.", llm_model="gemma3:12b", top_k=5)
    print(rag["answer"])

# 5. Save the updated database
veco.save_database(db_path, format="json")
print("Database updated and saved.")
