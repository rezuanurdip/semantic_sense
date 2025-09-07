import os
from sentence_transformers import SentenceTransformer


def load_local_model():
    """Load SentenceTransformer model packaged inside module."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_path, local_files_only=True)
