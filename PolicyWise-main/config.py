from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
EVALUATION_DIR = BASE_DIR / "evaluation"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 3

EMBEDDER_CHOICE = os.getenv(
    "EMBEDDER_CHOICE", "auto"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL", "gpt-4o-mini"
)