from veco import Vectorize
from pathlib import Path

# 1. Initialisiere Vektorisierer
veco = Vectorize()

# 2. Bestehende Vektordatenbank laden (wenn vorhanden)
db_path = "vector_db.json"
try:
    veco.load_database(db_path, format="json")
    print("Bestehende Datenbank geladen.")
except FileNotFoundError:
    print("Keine bestehende Datenbank gefunden. Neue wird erstellt.")

# 3. Verzeichnisschleife: Vektorisieren und ergänzen
folder = Path("test_data")
for file in folder.iterdir():
    if file.is_file():
        try:
            veco.vectorize(str(file), use_compression=False)
            print(f"{file.name} wurde erfolgreich vektorisiert.")
        except Exception as e:
            print(f"Fehler bei {file.name}: {e}")

# 4. Datenbank speichern
veco.save_database(db_path, format="json")
print("Datenbank aktualisiert und gespeichert.")
