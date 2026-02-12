# everbot_rag/embeddings.py
from __future__ import annotations
import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

_G_EMB = None

def get_emb_model(model_name: str) -> SentenceTransformer:
    global _G_EMB
    if _G_EMB is None:
        _G_EMB = SentenceTransformer(model_name)
    return _G_EMB

def embed_passages(model_name: str, passages: List[str]) -> np.ndarray:
    model = get_emb_model(model_name)
    embs = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embs, dtype="float32")

def embed_query(model_name: str, query: str) -> np.ndarray:
    model = get_emb_model(model_name)
    embs = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")
