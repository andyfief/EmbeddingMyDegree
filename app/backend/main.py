"""
main.py — FastAPI backend for School Search.

Environment variables (set before running):
    OPENAI_API_KEY   — required
    S3_BUCKET        — required  (e.g. andrewfief-embedding-school)
    SCHOOL_ROOT      — required  (e.g. C:\\Users\\FiefA\\Desktop\\School)
    S3_KEY           — optional  (default: chunks_embedded.jsonl)
    DYNAMO_TABLE     — optional  (default: search_queries)
    AWS_REGION       — optional  (default: us-east-1)
"""

import os
import time
import subprocess
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from s3_loader    import load_from_s3
from searcher     import embed_query, search, build_preview
from dynamo_logger import log_query

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
S3_BUCKET      = os.environ["S3_BUCKET"]
S3_KEY         = os.environ.get("S3_KEY",       "chunks_embedded.jsonl")
SCHOOL_ROOT    = os.environ["SCHOOL_ROOT"]
DYNAMO_TABLE   = os.environ.get("DYNAMO_TABLE",  "search_queries")
AWS_REGION     = os.environ.get("AWS_REGION",    "us-east-1")

# ---------------------------------------------------------------------------
# App state — populated at startup
# ---------------------------------------------------------------------------

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks, matrix = load_from_s3(S3_BUCKET, S3_KEY, AWS_REGION)
    app_state["chunks"] = chunks
    app_state["matrix"] = matrix
    print(f"[startup] Ready. {len(chunks):,} chunks loaded.")
    yield
    app_state.clear()


app = FastAPI(title="School Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int     = Field(5, ge=1, le=50)

class ChunkResult(BaseModel):
    rank:         int
    score:        float
    file_path:    str
    chunk_index:  int
    total_chunks: int
    category:     str
    preview:      str
    start_page:   int | None = None

class SearchResponse(BaseModel):
    query:      str
    k:          int
    latency_ms: float
    results:    list[ChunkResult]

class OpenRequest(BaseModel):
    file_path:  str
    start_page: int | None = None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(app_state.get("chunks", []))}


@app.post("/search", response_model=SearchResponse)
def search_route(req: SearchRequest):
    t0     = time.perf_counter()
    chunks = app_state["chunks"]
    matrix = app_state["matrix"]

    query_vec = embed_query(req.query, OPENAI_API_KEY)
    hits      = search(query_vec, matrix, req.k)

    results = []
    for rank, (idx, score) in enumerate(hits, 1):
        c = chunks[idx]
        results.append(ChunkResult(
            rank         = rank,
            score        = score,
            file_path    = c.get("file_path", ""),
            chunk_index  = c.get("chunk_index", 0),
            total_chunks = c.get("total_chunks", 1),
            category     = c.get("category", ""),
            preview      = build_preview(c),
            start_page   = c.get("start_page", None),
        ))

    latency_ms = (time.perf_counter() - t0) * 1000

    log_query(
        query      = req.query,
        k          = req.k,
        latency_ms = latency_ms,
        results    = [r.model_dump() for r in results],
        table_name = DYNAMO_TABLE,
        region     = AWS_REGION,
    )

    return SearchResponse(
        query      = req.query,
        k          = req.k,
        latency_ms = round(latency_ms, 2),
        results    = results,
    )


@app.post("/open")
def open_file(req: OpenRequest):
    abs_path = Path(SCHOOL_ROOT) / req.file_path

    if not abs_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")

    ext = abs_path.suffix.lower()

    # PDFs with a page: launch Chrome directly so #page=N fragment is preserved.
    # webbrowser.open and cmd start both drop the fragment on Windows.
    if ext == ".pdf" and req.start_page:
        uri = abs_path.as_uri() + f"#page={req.start_page}"
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        chrome = next((p for p in chrome_paths if os.path.exists(p)), None)
        if chrome:
            subprocess.Popen([chrome, uri])
        else:
            # Fallback: open without page if Chrome not found
            os.startfile(str(abs_path))
    else:
        os.startfile(str(abs_path))

    return {"opened": str(abs_path), "page": req.start_page}