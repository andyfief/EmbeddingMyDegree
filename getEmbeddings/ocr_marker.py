#!/usr/bin/env python3
"""
ocr_marker.py — OCR image-based PDF pages using marker-pdf (GPU-accelerated).
Reads ocr_candidates.txt, checks ingest_errors.log for which pages need OCR,
and outputs marker_chunks.jsonl in the same format as chunks.jsonl.

Setup:
    pip install marker-pdf
    (PyTorch with CUDA must already be installed)

Usage:
    python ocr_marker.py
"""

import re
import json
from pathlib import Path
from collections import defaultdict

# ===========================================================================
# CONFIG
# ===========================================================================

SCHOOL_ROOT    = "../../School"
LOG_FILE       = "ingest_errors.log"
OCR_CANDIDATES = "ocr_candidates.txt"
OUTPUT_JSONL   = "marker_chunks.jsonl"

# Chunking constants — must match ingest.py exactly
CHUNK_CHARS   = 6_000
OVERLAP_CHARS = 400
HEADER_CHARS  = 1_000

# ===========================================================================


# ---------------------------------------------------------------------------
# Error log parsing
# ---------------------------------------------------------------------------

def find_skipped_pages(log_path: Path) -> dict[str, list[int]]:
    """Returns {rel_path: [page_nums]} for SKIP_PAGE entries (1-indexed)."""
    skipped = defaultdict(list)
    pattern = re.compile(r"^SKIP_PAGE (.+?) page (\d+):")
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                skipped[m.group(1)].append(int(m.group(2)))
    return skipped


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def load_done_files(out_path: Path) -> set[str]:
    """Returns set of rel_paths already fully written to output."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                fp = obj.get("file_path")
                if fp:
                    done.add(fp)
            except json.JSONDecodeError:
                pass
    return done


# ---------------------------------------------------------------------------
# Chunking — matches ingest.py make_char_chunks exactly
# ---------------------------------------------------------------------------

def make_char_chunks(text: str, file_path: str, start_page: int,
                     start_chunk_idx: int) -> list[dict]:
    if not text.strip():
        return []

    header_text = text[:HEADER_CHARS]
    step        = CHUNK_CHARS - OVERLAP_CHARS
    starts      = list(range(0, len(text), step)) or [0]

    chunks = []
    for i, start in enumerate(starts):
        end        = min(start + CHUNK_CHARS, len(text))
        body       = text[start:end]
        chunk_text = f"path: {file_path}\nheader:\n{header_text}\nbody:\n{body}"
        chunks.append({
            "file_path":    file_path,
            "ext":          ".pdf",
            "category":     "document",
            "chunk_index":  start_chunk_idx + i,
            "total_chunks": -1,       # patched after all pages are processed
            "start_char":   start,
            "end_char":     end,
            "start_page":   start_page,
            "ocr":          True,
            "ocr_engine":   "marker",
            "text":         chunk_text,
        })
    return chunks


# ---------------------------------------------------------------------------
# marker conversion
# ---------------------------------------------------------------------------

def convert_pages(abs_path: Path, page_nums: list[int], artifact_dict) -> dict[int, str]:
    """
    Runs marker on each page individually using markdown output mode.
    page_nums is 1-indexed. Returns {page_num: markdown_text}.

    Processing one page at a time guarantees correct page attribution
    and uses marker's primary/best-supported output path.
    """
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.output import text_from_rendered

    page_texts: dict[int, str] = {}

    for page_num in sorted(page_nums):
        zero_idx = page_num - 1
        config = {
            "output_format": "markdown",
            "page_range": str(zero_idx),
            "force_ocr": True,
        }
        config_parser = ConfigParser(config)
        converter = PdfConverter(
            artifact_dict=artifact_dict,
            config=config_parser.generate_config_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )
        rendered  = converter(str(abs_path))
        text, _, _ = text_from_rendered(rendered)
        page_texts[page_num] = text.strip()

    return page_texts


# ---------------------------------------------------------------------------
# Process one PDF
# ---------------------------------------------------------------------------

def process_pdf(rel_path: str, page_nums: list[int], school_root: Path,
                out_path: Path, artifact_dict) -> int:

    abs_path = school_root / rel_path
    if not abs_path.exists():
        print(f"  MISSING: {abs_path}")
        return 0

    print(f"  Running marker on pages {sorted(page_nums)} ...")
    try:
        page_texts = convert_pages(abs_path, page_nums, artifact_dict)
    except Exception as e:
        print(f"  FAILED: {e}")
        return 0

    all_chunks = []
    for page_num in sorted(page_nums):
        text = page_texts.get(page_num, "").strip()
        if not text:
            print(f"    page {page_num} → empty")
            continue
        page_chunks = make_char_chunks(
            text, rel_path,
            start_page=page_num,
            start_chunk_idx=len(all_chunks),
        )
        all_chunks.extend(page_chunks)
        print(f"    page {page_num} → {len(page_chunks)} chunk(s), {len(text):,} chars")

    if not all_chunks:
        return 0

    # Patch total_chunks
    for chunk in all_chunks:
        chunk["total_chunks"] = len(all_chunks)

    with open(out_path, "a", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return len(all_chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from marker.models import create_model_dict

    school_root = Path(SCHOOL_ROOT)
    log_path    = Path(LOG_FILE)
    out_path    = Path(OUTPUT_JSONL)
    list_path   = Path(OCR_CANDIDATES)

    if not list_path.exists():
        print(f"ERROR: {list_path} not found. Run build_ocr_list.py first.")
        return
    if not log_path.exists():
        print(f"ERROR: {log_path} not found. Run ingest.py first.")
        return

    rel_paths     = [l.strip() for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    skipped_pages = find_skipped_pages(log_path)
    done_files    = load_done_files(out_path)

    # Build candidates — skip files already in output
    candidates: dict[str, list[int]] = {}
    for rel_path in rel_paths:
        if rel_path in done_files:
            continue
        abs_path = school_root / rel_path
        if not abs_path.exists():
            print(f"  WARNING: not found, skipping: {rel_path}")
            continue
        if rel_path in skipped_pages:
            pages = sorted(skipped_pages[rel_path])
        else:
            import fitz
            doc   = fitz.open(str(abs_path))
            pages = list(range(1, len(doc) + 1))
            doc.close()
        candidates[rel_path] = pages

    total_pages = sum(len(p) for p in candidates.values())
    print(f"\n  {len(candidates)} PDFs | {total_pages} pages to OCR")
    if len(done_files):
        print(f"  {len(done_files)} files already done (resume)")
    if total_pages == 0:
        print("  Nothing to do.")
        return

    confirm = input("\n  Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    # Load models once for the whole run
    print("\n  Loading marker models (first run downloads ~4GB)...")
    artifact_dict = create_model_dict()
    print("  Models loaded.\n")

    total_chunks = 0
    for rel_path, pages in sorted(candidates.items()):
        print(f"\n  [{rel_path}]  ({len(pages)} pages)")
        total_chunks += process_pdf(rel_path, pages, school_root, out_path, artifact_dict)

    print(f"\n{'='*60}")
    print(f"  Done. {total_chunks:,} chunks written to {out_path.resolve()}")
    print(f"  To embed: point embed_chunks.py INPUT_FILE at {OUTPUT_JSONL}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()