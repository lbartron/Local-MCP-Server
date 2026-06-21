from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL_NAME

# Load the model globally so it only boots up once when the app starts
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def generate_embedding(text: str) -> list[float]:
    # Pass input text to model so it can be encoded, convert returned 
    # numpy array into py list, then return list

    arr = model.encode(text)
    ret_list = arr.tolist()
    return ret_list
    