import shutil
from pathlib import Path

def clean_files(directories, keep=10):
    for dir_path in directories:
        try:
            # Ensure the directory exists
            path = Path(dir_path)
            if not path.is_dir():
                print(f"Skipping {dir_path}: Not a directory")
                continue

            # Get all files and folders in the directory, sorted by modification time
            items = sorted(
                path.iterdir(),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Retain the `keep` most recent files/folders
            for item in items[keep:]:
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"Deleted folder: {item}")
                else:
                    item.unlink()
                    print(f"Deleted file: {item}")
        except Exception as e:
            print(f"Error cleaning {dir_path}: {e}")

# Directories to clean
directories_to_clean = ["logs/", "vid/", "img/original", "img/annotated", "img/depth"]

# Run the cleaning function
clean_files(directories_to_clean)
