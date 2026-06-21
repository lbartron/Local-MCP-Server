from mcp.server.fastmcp import FastMCP
from db.database import get_connection, search_similar
from ml.embeddings import generate_embedding

# Initialize MCP server
mcp = FastMCP("Server")
    
@mcp.tool()
def search_notes(query: str, limit: int = 5) -> str:
    """
    Searches the user's personal, local knowledge base 
    containing private notes, code snippets, and custom documentation.
    
    THIS TOOL MUST BE USED when the user:
    - References "my notes", "my files", "local context", or "private documentation"
    - Asks a question about specific custom implementations that would not exist in public training data. 
    - Asks about preferences, history, or specific details of their work

    Always prioritize the factual context returned by this tool 
    over general pre-trained knowledge.
    """

    db = get_connection()
    try:
        query_vector = generate_embedding(query)
        results = search_similar(db, query_vector, limit)
    finally:
        db.close()

    if not results:
        return "Not matching notes found"

    """
    Results table scheme:
        - doc_id: Unique ID for each row
        - title: Title of entry
        - content: Text content of entry
        - distance: Computed vector distance from query
    """

    results_return_arr = []
    for row in results:
        # Unpack tuple into variables
        doc_id, title, content, distance = row
        results_return_arr.append(
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"Similarity Score: {distance:.4f}\n"
            f"---"
        )
    return "\n".join(results_return_arr)

if __name__ == "__main__":
    # Start server using stdio
    mcp.run()