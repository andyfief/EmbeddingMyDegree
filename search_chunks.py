#!/usr/bin/env python3
"""
search_chunks.py — Search your embedded JSONL using cosine similarity.

Usage:
    python search_chunks.py
"""

import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

# ===========================================================================
# CONFIG — edit these before each run
# ===========================================================================

INPUT_FILE = "chunks_embedded.jsonl"
TOP_K      = 5

# ===========================================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL      = "text-embedding-3-small"
DIMENSIONS = 1536

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_embedded(path: Path):
    chunks, vectors = [], []
    print(f"Loading {path} ...")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "embedding" not in obj:
                continue
            chunks.append(obj)
            vectors.append(obj["embedding"])

    matrix = np.array(vectors, dtype=np.float32)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-10, None)

    print(f"  Loaded {len(chunks):,} chunks  |  matrix shape: {matrix.shape}")
    return chunks, matrix

# ---------------------------------------------------------------------------
# Embed query
# ---------------------------------------------------------------------------

def embed_query(text: str) -> np.ndarray:
    response = client.embeddings.create(model=MODEL, input=[text], dimensions=DIMENSIONS)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query_vec: np.ndarray, matrix: np.ndarray, top_k: int):
    scores  = matrix @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results, chunks):
    print(f"\n{'='*70}")
    for rank, (idx, score) in enumerate(results, 1):
        c         = chunks[idx]
        file_path = c.get("file_path", "unknown")
        chunk_i   = c.get("chunk_index", 0)
        total     = c.get("total_chunks", 1)
        category  = c.get("category", "")
        text      = c.get("text", "")

        if "body:\n" in text:
            preview = text.split("body:\n", 1)[1].strip()
        elif "header:\n" in text:
            preview = text.split("header:\n", 1)[1].strip()
        else:
            preview = text.strip()
        preview = preview[:300].replace("\n", " ")

        print(f"\n  #{rank}  score={score:.4f}  [{category}]  chunk {chunk_i+1}/{total}")
        print(f"       {file_path}")
        print(f"       {preview}...")
    print(f"\n{'='*70}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chunks, matrix = load_embedded(Path(INPUT_FILE))
    QUERY      = input("Type Query")
    print(f"\nEmbedding query: '{QUERY}'")
    query_vec = embed_query(QUERY)
    results   = search(query_vec, matrix, TOP_K)
    print_results(results, chunks)


if __name__ == "__main__":
    main()