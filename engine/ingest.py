import os, time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from db.database import get_connection, insert_document_chunks, delete_document_by_path
from ml.embeddings import generate_embedding

def split_text(text: str) -> list[str]:
    """ Splits large documents into smaller pieces so embedding 
        model can full process each file
    """
    MIN_CHUNK_LENGTH = 10
    valid_chunks = []
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()
        if len(clean_paragraph) > MIN_CHUNK_LENGTH:
            valid_chunks.append(clean_paragraph)

    return valid_chunks

def process_single_file(file_path: Path):
    """ Ingests a single file into the database based on the provided file path """

    # Filter for only markdown or text files, and ignore temp files
    if not file_path.suffix in [".md", ".txt"] or file_path.suffix.startswith("."):
        return

    db = get_connection()
    try:
        normalized_file_path = str(file_path.resolve())
        os_modification_time = os.path.getmtime(normalized_file_path)

        # Wipe old chunks to prevent duplicates
        delete_document_by_path(db, file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except PermissionError:
            # In case text editor is still writing to file after user saves.
            # This prevents watchdog from crashing after instantly firing after
            # a file save. Wait 200ms to retry
            time.sleep(0.2)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        chunks = split_text(content)
        if not chunks: return

        # Create embeddings for gathered text
        vectors = [generate_embedding(chunk) for chunk in chunks]
            
        # Insert text and embeddings into db
        title = file_path.name
        insert_document_chunks(db, title, normalized_file_path, os_modification_time, chunks, vectors)
    finally:
        db.close()

def startup_file_sync(db, directory_path: Path):
    """ Syncs local file system notes with SQLite vector database """
    pass

class NotesUpdateHandler(FileSystemEventHandler):
    def on_created(event): 
        pass

    def on_modified(event): 
        # Call process_single_file().
        pass

    def on_deleted(event): 
        # Call delete_document_by_path().
        pass

def start_watchdog():

    pass