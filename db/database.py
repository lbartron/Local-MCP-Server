import sqlite3
import sqlite_vec
from core.config import DB_PATH, EMBEDDING_DIMENSIONS

def get_connection():
    """ Create and return an SQLite connection with sqlite-vec loaded """

    db = sqlite3.connect(DB_PATH)

    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    return db

def init_tables(connection):
    """ Create the text table and the vec0 virtual table """

    cursor = connection.cursor()
    # Text Table Creation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            file_path TEXT NOT NULL,
            last_modified REAL NOT NULL
        );
    """)

    # Vec0 Virtual Table Creation
    # Update float depending on dimensions of embedding model
    # Currently all-MiniLM-L6-v2 uses 384 dimensions
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIMENSIONS}]
        );
    """)
    connection.commit()
    
def insert_document(connection, title: str, content: str, file_path: str, last_modified: float, vector: list[float]):
    """ Insert text into documents table, and its vector into vec_documents. 
        This should only be called for small files as embedding model can only
        process small chunks of data at a time. Use insert_document_chunks for 
        larger file ingestion
    """
    cursor = connection.cursor()
    
    # Insert into the documents table
    cursor.execute("""
        INSERT INTO documents (title, content, file_path, last_modified) 
        VALUES (?, ?, ?, ?)
    """, (title, content, file_path, last_modified))
    doc_id = cursor.lastrowid # Get auto-generated ID for vec_documents
    
    # Serialize the vector and insert into the virtual table with the matching ID
    serialized_vector = sqlite_vec.serialize_float32(vector)
    cursor.execute("INSERT INTO vec_documents (id, embedding) VALUES (?, ?)", (doc_id, serialized_vector))
    
    connection.commit()
    return doc_id

def insert_document_chunks(connection, title: str, file_path: str, last_modified: float, chunks: list[str], vectors: list[list[float]]):
    """ Insert an array of text chunks and corresponding vectors 
        into documents and vec_documents tables 
    """
    cursor = connection.cursor()

    for chunk, vector in zip(chunks, vectors):
        # Insert into the documents table
        cursor.execute("""
            INSERT INTO documents (title, content, file_path, last_modified) 
            VALUES (?, ?, ?, ?)
        """, (title, chunk, file_path, last_modified))
        doc_id = cursor.lastrowid # Get auto-generated ID for vec_documents
        
        # Serialize the vector and insert into the virtual table with the matching ID
        serialized_vector = sqlite_vec.serialize_float32(vector)
        cursor.execute("INSERT INTO vec_documents (id, embedding) VALUES (?, ?)", (doc_id, serialized_vector))

    connection.commit()

def delete_document_by_path(connection, file_path):
    """ Delete database entries associated with a specific file path """
    cursor = connection.cursor()
    
    # Delete from the virtual vector table
    # Use subquery to find all chunk IDs associated with this file path
    cursor.execute("""
        DELETE FROM vec_documents 
        WHERE id IN (
            SELECT id FROM documents WHERE file_path = ?
        )
    """, (file_path,))
    
    # Delete from the documents table
    cursor.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
    
    connection.commit()

def search_similar(connection, vector: list[float], limit: int = 5,):
    """ Perform L2 distance query and return matching text chunks """
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
        WHERE vec_documents.embedding MATCH ? AND k = ?
        ORDER BY distance
    """, (query_vector, limit))
    
    return cursor.fetchall()
