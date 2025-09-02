from veco import Vectorize
from pathlib import Path

# 1. Initialize the vectorizer
veco = Vectorize()

# 2. Load existing vector database (if available)
db_path = "vector_db.json"
try:
    veco.load_database(db_path, format="json")
    print("Existing database loaded.")
except FileNotFoundError:
    print("No existing database found. A new one will be created.")

# 3. Access the 'test_data' folder in the parent directory
# Get the current script's directory (where veco_test.py is located)
current_directory = Path(__file__).resolve().parent
print(f"Current Directory: {current_directory}")

# Navigate to the parent folder
parent_directory = current_directory.parent
print(f"Parent Directory: {parent_directory}")

# Access the 'test_data' folder in the parent directory
test_data_directory = parent_directory / 'test_data'
print(f"Test Data Directory: {test_data_directory}")

# 4. Directory loop: Vectorize and append data
# Loop through all files in the 'test_data' folder
if test_data_directory.exists() and test_data_directory.is_dir():
    for file in test_data_directory.iterdir():
        if file.is_file():
            try:
                veco.vectorize(str(file), use_compression=False)
                print(f"{file.name} was successfully vectorized.")
            except Exception as e:
                print(f"Error with {file.name}: {e}")
else:
    print(f"Test data directory does not exist: {test_data_directory}")

# 5. Save the updated database
veco.save_database(db_path, format="json")
print("Database updated and saved.")
