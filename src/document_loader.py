import glob
import os

def load_chunks(directory: str) -> list[str]:
    """
    Loads all .txt files from a given directory into a list of strings.
    
    :param directory: Directory containing .txt chunks.
    :return: A list of document texts.
    """
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    docs = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            print(f"DEBUG: Loaded {len(text)} chars from {path}")
            docs.append(text)
    return docs
