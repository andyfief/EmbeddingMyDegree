"""
s3_loader.py — Download chunks_embedded.jsonl from S3 and build numpy search matrix.
"""

import json
import tempfile
import numpy as np
import boto3
from pathlib import Path


def load_from_s3(bucket: str, key: str, region: str) -> tuple[list[dict], np.ndarray]:
    """
    Downloads the JSONL from S3, parses every line with an embedding,
    and returns (chunks, L2-normalised float32 matrix).
    Matrix rows are unit vectors so dot product == cosine similarity.
    """
    print(f"[s3_loader] Downloading s3://{bucket}/{key} ...")
    s3 = boto3.client("s3", region_name=region)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        s3.download_fileobj(bucket, key, tmp)
        tmp_path = Path(tmp.name)

    print(f"[s3_loader] Download complete. Parsing ...")
    chunks, vectors = [], []

    with open(tmp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "embedding" not in obj:
                continue
            chunks.append(obj)
            vectors.append(obj["embedding"])

    tmp_path.unlink(missing_ok=True)

    if not vectors:
        raise RuntimeError("No embedded chunks found in the downloaded JSONL.")

    matrix = np.array(vectors, dtype=np.float32)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-10, None)

    print(f"[s3_loader] Loaded {len(chunks):,} chunks | matrix shape: {matrix.shape}")
    return chunks, matrix
