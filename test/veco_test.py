from pathlib import Path
from VeCo import Vectorize

def main():
    # Absoluter Projekt-Root als Path
    project_root = Path(r"X:\01_Dokumente\12_Startup-Mentor\MP_Data")

    db_path = project_root / "vector_db.json"
    test_data_directory = project_root / "test_data"

    print(f"Project root:        {project_root}")
    print(f"JSON DB:             {db_path}")
    print(f"Test data directory: {test_data_directory}")

    # Vectorizer initialisieren
    veco = Vectorize(preload_json_path=str(db_path))

    # Datenbank laden
    veco.load_database(str(db_path))
    print("Database loaded (or initialized empty).")

    # Dateien im test_data-Ordner vektorisieren
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

    # DB speichern
    veco.save_database(str(db_path))
    print(f"Database updated and saved → {db_path}")

    veco.close()

if __name__ == "__main__":
    main()
