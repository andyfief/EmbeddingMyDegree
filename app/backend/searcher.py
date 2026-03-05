"""
searcher.py — Embed a query and run cosine search against the preloaded matrix.
"""

import numpy as np
from openai import OpenAI

MODEL      = "text-embedding-3-small"
DIMENSIONS = 1536

_client: OpenAI | None = None


def get_client(api_key: str) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client


def embed_query(query: str, api_key: str) -> np.ndarray:
    client = get_client(api_key)
    response = client.embeddings.create(model=MODEL, input=[query], dimensions=DIMENSIONS)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def search(query_vec: np.ndarray, matrix: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    scores  = matrix @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def build_preview(chunk: dict, max_chars: int = 300) -> str:
    text = chunk.get("text", "")
    if "body:\n" in text:
        body = text.split("body:\n", 1)[1].strip()
    elif "header:\n" in text:
        body = text.split("header:\n", 1)[1].strip()
    else:
        body = text.strip()
    return body[:max_chars].replace("\n", " ")
