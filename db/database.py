import sqlite3
import sqlite_vec
from core.config import DB_PATH, EMBEDDING_DIMENSIONS

def get_connection():
    """Create and return an SQLite connection with sqlite-vec loaded."""
    db = sqlite3.connect(DB_PATH)

    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    return db

def init_tables(connection):
    """Create the text table and the vec0 virtual table."""
    cursor = connection.cursor()
    # Text Table Creation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL
        );
    """)

    # Vec0 Virtual Table Creation
    # Update float depending on dimensions of embedding model
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIMENSIONS}]
        );
    """)
    connection.commit()
    

def search_similar(connection, vector: list[float], limit: int = 5,):
    """Perform L2 distance query and return matching text chunks."""
    cursor = connection.cursor()

    # Convert Python list to serialized format for query
    query_vector = sqlite_vec.serialize_float32(vector)

    # KNN Query
    cursor.execute("""
        SELECT 
            documents.id,
            documents.title,
            documents.content,
            vec_documents.distance
        FROM vec_documents
        LEFT JOIN documents ON vec_documents.id = documents.id
        WHERE vec_documents.embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (query_vector, limit))
    
    return cursor.fetchall()
