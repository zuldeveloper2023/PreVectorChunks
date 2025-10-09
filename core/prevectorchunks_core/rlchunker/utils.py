import re
from sentence_transformers import SentenceTransformer

_embed_model = None

def split_sentences(text):
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s]

def get_embed_model(name='all-MiniLM-L6-v2'):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(name)
    return _embed_model

def get_embeddings(sentences):
    model = get_embed_model()
    return model.encode(sentences, convert_to_tensor=True)
