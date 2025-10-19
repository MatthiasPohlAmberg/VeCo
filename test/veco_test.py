from pathlib import Path

from VeCo import Vectorize


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "test" / "vector_db.json"
    test_data_directory = project_root / "test_data"

    print(f"Project root:        {project_root}")
    print(f"JSON DB:             {db_path}")
    print(f"Test data directory: {test_data_directory}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    veco = Vectorize(preload_json_path=str(db_path))

    try:
        veco.load_database(str(db_path))
        print("Database loaded (or initialized empty).")

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

        veco.save_database(str(db_path))
        print(f"Database updated and saved → {db_path}")
    finally:
        veco.close()


if __name__ == "__main__":
    main()
