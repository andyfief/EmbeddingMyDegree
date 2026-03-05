#!/usr/bin/env python3
"""
ocr_pdf_pages.py — Supplement for ingest.py.
Finds PDFs that had image-based pages skipped during ingest, runs GPT-4o vision
OCR on those pages, and appends the resulting chunks to chunks.jsonl.

Requirements:
    pip install pymupdf openai pillow
    export OPENAI_API_KEY=sk-...

Usage:
    # First run ingest.py normally. Then:
    python ocr_pdf_pages.py

    # Or target a single PDF directly:
    python ocr_pdf_pages.py --file "path/to/some/lecture.pdf"
"""

import os
import re
import json
import base64
import time
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# ===========================================================================
# CONFIG — edit before running
# ===========================================================================

SCHOOL_ROOT      = "../../School"
CHUNKS_JSONL     = "chunks.jsonl"
LOG_FILE         = "ingest_errors.log"
OCR_OUTPUT_JSONL = "chunks.jsonl"
OCR_CANDIDATES   = "ocr_candidates.txt"

DPI              = 150    # render resolution. 150 is fast & usually enough; use 200 for dense math
MAX_TOKENS       = 2000   # GPT-4o tokens per page — enough for a dense slide or textbook page
PRICE_INPUT_1M   = 2.50   # $/1M input tokens  (gpt-4o, as of 2025)
PRICE_OUTPUT_1M  = 10.00  # $/1M output tokens
SKIP_CONFIRM     = False  # True = run without cost confirmation

# Ingest chunking constants (keep in sync with ingest.py)
CHUNK_CHARS  = 6_000
OVERLAP_CHARS = 400
HEADER_CHARS = 1_000

# ===========================================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Step 1 — Find which PDFs had pages skipped
# ---------------------------------------------------------------------------

def find_skipped_pages(log_path: Path) -> dict[str, list[int]]:
    """
    Parse ingest_errors.log for lines like:
        SKIP_PAGE some/file.pdf page 3: only 5 non-empty lines (image-based?)
    Returns {rel_path: [page_numbers]} (1-indexed).
    """
    skipped: dict[str, list[int]] = defaultdict(list)
    if not log_path.exists():
        return skipped
    pattern = re.compile(r"^SKIP_PAGE (.+?) page (\d+):")
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                rel_path = m.group(1)
                page_num = int(m.group(2))
                skipped[rel_path].append(page_num)
    return skipped


def find_wholly_skipped(log_path: Path) -> set[str]:
    """Files skipped entirely (SKIP_FILE entries) — these need full OCR."""
    skipped = set()
    if not log_path.exists():
        return skipped
    pattern = re.compile(r"^SKIP_FILE (.+?):")
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                skipped.add(m.group(1))
    return skipped


# ---------------------------------------------------------------------------
# Step 2 — Render pages to images
# ---------------------------------------------------------------------------

def render_page_to_base64(pdf_path: Path, page_num: int, dpi: int = DPI) -> str:
    """
    Renders a single PDF page to a JPEG and returns base64.
    page_num is 1-indexed.
    Uses pymupdf (fitz) — no poppler needed.
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]  # fitz uses 0-indexed
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 DPI is fitz default
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    return base64.standard_b64encode(img_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Step 3 — OCR a page with GPT-4o vision
# ---------------------------------------------------------------------------

OCR_SYSTEM_PROMPT = """You are an OCR engine. The user sends you an image of a single page from an academic PDF.
Your job is to extract ALL text from the image as faithfully as possible.

Rules:
- Output ONLY the extracted text, nothing else.
- Preserve the logical reading order (top to bottom, left to right for English).
- Keep headings, bullet points, numbered lists, and code blocks intact.
- For math: use LaTeX inline notation like $x^2$ or $$\\sum_{i=0}^n i$$
- For tables: reproduce them as plain text with spacing or | separators.
- If the page is mostly a diagram/figure with little text, output the caption and any visible labels.
- Do not describe images. Do not add commentary. Just the text."""


def ocr_page(base64_image: str, file_path: str, page_num: int) -> str:
    """Call GPT-4o vision to extract text from a rendered page."""
    max_retries = 4
    wait = 10

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Extract all text from this page (file: {file_path}, page {page_num})."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            err = str(e)
            if "400" in err:
                raise  # bad request, don't retry
            if attempt == max_retries - 1:
                raise
            print(f"\n  API error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, 60)

    return ""


# ---------------------------------------------------------------------------
# Step 4 — Chunk OCR'd text (same logic as ingest.py make_char_chunks)
# ---------------------------------------------------------------------------

def make_char_chunks(text: str, file_path: str, ext: str, start_chunk_idx: int,
                     page_num: int | None = None) -> list[dict]:
    """Character-based chunking, matching ingest.py exactly."""
    if not text.strip():
        return []

    header_text = text[:HEADER_CHARS]
    step = CHUNK_CHARS - OVERLAP_CHARS
    starts = list(range(0, len(text), step)) or [0]

    chunks = []
    for i, start in enumerate(starts):
        end  = min(start + CHUNK_CHARS, len(text))
        body = text[start:end]
        chunk_text = f"path: {file_path}\nheader:\n{header_text}\nbody:\n{body}"
        chunk = {
            "file_path":    file_path,
            "ext":          ext,
            "category":     "document",
            "chunk_index":  start_chunk_idx + i,
            "total_chunks": -1,          # patched after all pages processed
            "start_char":   start,
            "end_char":     end,
            "ocr":          True,        # flag so you can tell these apart later
            "text":         chunk_text,
        }
        if page_num is not None:
            chunk["start_page"] = page_num
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Step 5 — Get existing chunk count for a file (so we don't collide indices)
# ---------------------------------------------------------------------------

def existing_chunk_count(chunks_path: Path, rel_path: str) -> int:
    """Count how many chunks for this rel_path already exist in the JSONL."""
    count = 0
    if not chunks_path.exists():
        return 0
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("file_path") == rel_path:
                    count += 1
            except json.JSONDecodeError:
                pass
    return count


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(n_pages: int) -> None:
    # GPT-4o vision: high detail = ~765 tokens input per tile + 85 base
    # A typical A4 page at 150dpi ≈ 1240x1754px → ~3 tiles → ~2,380 input tokens
    # Plus MAX_TOKENS output tokens per page
    est_input_per_page  = 2_500
    est_output_per_page = MAX_TOKENS
    total_input  = n_pages * est_input_per_page
    total_output = n_pages * est_output_per_page
    cost = (total_input / 1_000_000) * PRICE_INPUT_1M + (total_output / 1_000_000) * PRICE_OUTPUT_1M
    print(f"\n  Pages to OCR     : {n_pages:,}")
    print(f"  Est. input tokens: ~{total_input:,}")
    print(f"  Est. cost        : ~${cost:.2f}  (gpt-4o standard)")
    print(f"  Note: actual cost varies; very dense pages cost more.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_pdf(rel_path: str, page_nums: list[int], school_root: Path,
                out_path: Path, verbose: bool = True) -> int:
    """OCR the given pages of a PDF and append chunks to out_path. Returns chunk count."""
    abs_path = school_root / rel_path
    if not abs_path.exists():
        print(f"  MISSING: {abs_path}")
        return 0

    all_new_chunks = []
    start_idx      = existing_chunk_count(out_path, rel_path)

    for page_num in sorted(page_nums):
        if verbose:
            print(f"  OCR page {page_num} of {rel_path} ...", end=" ", flush=True)
        try:
            b64 = render_page_to_base64(abs_path, page_num)
            text = ocr_page(b64, rel_path, page_num)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        page_chunks = make_char_chunks(
            text, rel_path, ".pdf",
            start_chunk_idx=start_idx + len(all_new_chunks),
            page_num=page_num,
        )
        all_new_chunks.extend(page_chunks)
        if verbose:
            print(f"→ {len(page_chunks)} chunk(s), {len(text):,} chars")

    if not all_new_chunks:
        return 0

    # Patch total_chunks now that we know the count
    total = start_idx + len(all_new_chunks)
    for chunk in all_new_chunks:
        chunk["total_chunks"] = total

    with open(out_path, "a", encoding="utf-8") as f:
        for chunk in all_new_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return len(all_new_chunks)


def main():
    import fitz
    school_root = Path(SCHOOL_ROOT)
    log_path    = Path(LOG_FILE)
    out_path    = Path(OCR_OUTPUT_JSONL)
    list_path   = Path(OCR_CANDIDATES)

    if not list_path.exists():
        print(f"ERROR: {list_path} not found. Run build_ocr_list.py first.")
        return

    rel_paths     = [l.strip() for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    skipped_pages = find_skipped_pages(log_path)
    wholly_skipped = find_wholly_skipped(log_path)

    candidates = {}
    for rel_path in rel_paths:
        abs_path = school_root / rel_path
        if not abs_path.exists():
            print(f"  WARNING: not found, skipping: {rel_path}")
            continue
        if rel_path in skipped_pages:
            pages = sorted(skipped_pages[rel_path])
        else:
            doc = fitz.open(str(abs_path))
            pages = list(range(1, len(doc) + 1))
            doc.close()
        candidates[rel_path] = pages

    total_pages = sum(len(p) for p in candidates.values())
    print(f"  {len(candidates)} PDFs, {total_pages} pages to OCR")
    estimate_cost(total_pages)

    confirm = input("  Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    total_chunks = 0
    for rel_path, pages in sorted(candidates.items()):
        print(f"\n  [{rel_path}]  ({len(pages)} pages: {pages})")
        total_chunks += process_pdf(rel_path, pages, school_root, out_path)

    print(f"\n{'='*60}")
    print(f"  OCR complete. {total_chunks:,} new chunks appended to {out_path.resolve()}")
    print(f"  Re-run embed_chunks.py with RESUME=True to embed the new chunks.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()