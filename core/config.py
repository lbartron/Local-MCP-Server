import os
from pathlib import Path
from dotenv import load_dotenv

# Load user variables from the .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

_env_notes_dir = os.getenv("NOTES_DIRECTORY")

if _env_notes_dir:
    # Use user provided path
    NOTES_DIRECTORY = Path(_env_notes_dir)
else:
    # If not provided, generate 'default_notes' folder in project directory
    NOTES_DIRECTORY = BASE_DIR / "default_notes"
    NOTES_DIRECTORY.mkdir(exist_ok=True)

# Database and Model Configs
DB_PATH = BASE_DIR / "notes_vectors.db"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = 384