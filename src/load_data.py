from pathlib import Path


def load_texts(folder):
    """Read every .txt file in a folder and return {file_name: text}."""
    texts = {}
    for path in sorted(Path(folder).glob("*.txt")):
        texts[path.stem] = path.read_text(encoding="utf-8")
    return texts


if __name__ == "__main__":
    jds = load_texts("data/jds")
    resumes = load_texts("data/resumes")
    print(len(jds), "JDs and", len(resumes), "resumes loaded")
    print(list(jds.keys()))