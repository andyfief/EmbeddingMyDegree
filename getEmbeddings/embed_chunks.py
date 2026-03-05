#!/usr/bin/env python3
"""
embed_chunks.py — Embed a JSONL of text chunks using OpenAI text-embedding-3-small.

Setup:
    pip install openai tqdm
    export OPENAI_API_KEY=sk-...
    python embed_chunks.py
"""

import os
import json
import time
import tiktoken
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ===========================================================================
# RUN CONFIG — edit these before each run
# ===========================================================================

INPUT_FILE  = "marker_chunks.jsonl"            # Input JSONL of chunks
OUTPUT_FILE = "chunks_embedded.jsonl"   # Output JSONL with embeddings added

LIMIT = 'ALL'          # Max chunks to embed. Set to "ALL" to embed everything.

SKIP_CONFIRM = False  # True  = just run without asking
                      # False = print cost estimate and wait for y/N

RESUME = True         # True  = skip chunks already present in OUTPUT_FILE
                      # False = start fresh (will overwrite output file)

# ===========================================================================
# MODEL CONFIG — unlikely to need changing
# ===========================================================================

MODEL        = "text-embedding-3-small"
DIMENSIONS   = 1536   # Native dim for text-embedding-3-small
BATCH_SIZE   = 20     # Chunks per API call — keeps requests well under 40k TPM limit
PRICE_PER_1M = 0.02   # $/1M tokens — standard API (Batch API is $0.01/1M)

# ===========================================================================

client  = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
encoder = tiktoken.get_encoding("cl100k_base")  # encoding used by text-embedding-3-small

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_texts(chunk: dict) -> list[str]:
    """
    Builds the string(s) to embed from a chunk.
    Uses tiktoken to count exact tokens so we never exceed the 8192 token limit.
    Splits the body into overlapping sub-chunks if needed — no data is lost.
    """
    MAX_TOKENS    = 8000  # leave a 192 token buffer under the 8192 hard limit
    OVERLAP_TOKENS = 100

    file_path = chunk.get("file_path", "")
    raw = chunk.get("text", "").strip()

    if "header:\n" in raw and "body:\n" in raw:
        header_part = raw.split("header:\n", 1)[1].split("body:\n")[0].strip()
        body_part   = raw.split("body:\n", 1)[1].strip()
        content = body_part if header_part in body_part else f"{header_part}\n\n{body_part}"
    elif "header:\n" in raw:
        content = raw.split("header:\n", 1)[1].strip()
    elif "body:\n" in raw:
        content = raw.split("body:\n", 1)[1].strip()
    else:
        content = raw

    prefix     = f"FILE: {file_path}\n\n" if file_path else ""
    prefix_toks = encoder.encode(prefix)
    body_toks   = encoder.encode(content)
    body_limit  = MAX_TOKENS - len(prefix_toks)

    if len(body_toks) <= body_limit:
        return [prefix + content]

    # Split body token list into overlapping windows, decode each back to text
    print(f"\n  ⚠️  Oversized chunk split ({len(body_toks):,} tokens): {file_path}")
    step = body_limit - OVERLAP_TOKENS
    parts = []
    for start in range(0, len(body_toks), step):
        window = body_toks[start:start + body_limit]
        parts.append(prefix + encoder.decode(window))
    return parts

# ---------------------------------------------------------------------------
# Embedding — exponential backoff for rate limits
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Sends a batch to OpenAI and returns vectors in input order.
    On rate limit (429) or transient errors, retries with exponential backoff:
        15s → 30s → 60s → 120s → 120s → give up
    """
    max_retries = 6
    wait = 15

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=MODEL,
                input=texts,
                dimensions=DIMENSIONS,
            )
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

        except Exception as e:
            if attempt == max_retries - 1:
                raise

            err_str = str(e)
            is_rate_limit  = "429" in err_str or "rate_limit" in err_str.lower()
            is_bad_request = "400" in err_str  # e.g. token limit — retrying won't help

            if is_bad_request:
                raise  # fail fast, don't waste retry budget

            reason = "Rate limit hit" if is_rate_limit else f"API error: {e}"
            print(f"\n  {reason}. Waiting {wait}s (retry {attempt + 1}/{max_retries - 1})...")
            time.sleep(wait)
            wait = min(wait * 2, 120)

# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

def estimate_cost(chunks: list[dict]) -> None:
    all_texts     = [t for c in chunks for t in extract_texts(c)]
    total_tokens  = sum(len(encoder.encode(t)) for t in all_texts)
    approx_cost   = (total_tokens / 1_000_000) * PRICE_PER_1M

    print(f"\n  Chunks to embed  : {len(chunks):,}")
    print(f"  Texts after split: {len(all_texts):,}")
    print(f"  Exact tokens     : {total_tokens:,}")
    print(f"  Estimated cost   : ${approx_cost:.4f}  (standard API)")
    print(f"  Estimated cost   : ${approx_cost / 2:.4f}  (Batch API — async, 24hr)")
    print()

# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def chunk_id(chunk: dict) -> str:
    ocr = ":ocr" if chunk.get("ocr") else ""
    return f"{chunk.get('file_path', '')}::{chunk.get('chunk_index', 0)}{ocr}"

def load_done_ids(output_path: Path) -> set[str]:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "embedding" in obj:
                    done.add(chunk_id(obj))
            except json.JSONDecodeError:
                pass
    return done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_path  = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    # Load all chunks
    print(f"Loading {input_path} ...")
    chunks = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks):,} chunks total.")

    # Resume or fresh start
    if RESUME:
        done_ids = load_done_ids(output_path)
        if done_ids:
            print(f"  Resuming — {len(done_ids):,} already embedded, skipping.")
        write_mode = "a"
    else:
        done_ids = set()
        write_mode = "w"
        print("  RESUME=False — starting fresh, output file will be overwritten.")

    pending = [c for c in chunks if chunk_id(c) not in done_ids]

    # Apply limit
    if LIMIT != "ALL":
        if len(pending) > LIMIT:
            print(f"  LIMIT={LIMIT} — capping at {LIMIT} of {len(pending):,} pending chunks.")
            pending = pending[:LIMIT]

    if not pending:
        print("  Nothing to embed.")
        return

    estimate_cost(pending)

    if not SKIP_CONFIRM:
        confirm = input("  Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return

    # Embed and write incrementally
    with open(output_path, write_mode, encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(pending), BATCH_SIZE), desc="Embedding"):
            batch = pending[i : i + BATCH_SIZE]

            # Each chunk may expand into multiple texts if it was oversized
            texts = []
            chunk_map = []  # one entry per text, pointing back to its source chunk
            for chunk in batch:
                for sub_text in extract_texts(chunk):
                    texts.append(sub_text)
                    chunk_map.append(chunk)

            try:
                vectors = embed_batch(texts)
            except Exception as e:
                print(f"\n  Fatal error on batch {i // BATCH_SIZE}: {e}")
                raise

            for chunk, vector in zip(chunk_map, vectors):
                out = dict(chunk)
                out["embedding"] = vector
                out_f.write(json.dumps(out) + "\n")

            out_f.flush()

    total_out = sum(1 for _ in open(output_path, encoding="utf-8"))
    print(f"\n  Done. {output_path} now has {total_out:,} embedded chunks.")


if __name__ == "__main__":
    main()