#!/usr/bin/env python3
"""
review_chunks.py — Interactive browser for chunks.jsonl (or chunks_embedded.jsonl).

Usage:
    python review_chunks.py                          # browse all chunks
    python review_chunks.py --file chunks.jsonl      # explicit file
    python review_chunks.py --ext .pdf --limit 20    # filter by extension
    python review_chunks.py --search "neural net"    # keyword filter (text search)
    python review_chunks.py --stats                  # print stats only, no browsing
    python review_chunks.py --export bad_chunks.txt  # dump all previews to a file

Controls during browsing:
    ENTER  → next chunk
    b      → back one chunk
    s      → skip 10 forward
    f      → jump to a specific chunk index
    q      → quit
"""

import json
import argparse
import textwrap
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "chunks.jsonl"      # falls back to chunks_embedded.jsonl if missing
PREVIEW_CHARS = 1_200               # how many chars of body to show per chunk
WRAP_WIDTH    = 100                 # console wrap width

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_chunks(path: Path, ext_filter: str | None, search: str | None) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if ext_filter and obj.get("ext", "").lower() != ext_filter.lower():
                continue

            if search:
                text = obj.get("text", "").lower()
                if search.lower() not in text:
                    continue

            chunks.append(obj)
    return chunks

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(chunks: list[dict]) -> None:
    ext_counts      = defaultdict(int)
    cat_counts      = defaultdict(int)
    file_counts     = defaultdict(int)
    char_lens       = []
    empty_chunks    = 0

    for c in chunks:
        ext_counts[c.get("ext", "??")] += 1
        cat_counts[c.get("category", "??")] += 1
        file_counts[c.get("file_path", "??")] += 1
        text = c.get("text", "")
        body = text.split("body:\n", 1)[1] if "body:\n" in text else text
        body_chars = len(body.strip())
        char_lens.append(body_chars)
        if body_chars == 0:
            empty_chunks += 1

    print(f"\n{'='*60}")
    print(f"  Total chunks       : {len(chunks):,}")
    print(f"  Unique files       : {len(file_counts):,}")
    print(f"  Empty body chunks  : {empty_chunks:,}")

    if char_lens:
        avg  = sum(char_lens) / len(char_lens)
        mn   = min(char_lens)
        mx   = max(char_lens)
        p50  = sorted(char_lens)[len(char_lens)//2]
        print(f"\n  Body length (chars)")
        print(f"    min    : {mn:,}")
        print(f"    median : {p50:,}")
        print(f"    avg    : {avg:,.0f}")
        print(f"    max    : {mx:,}")

    print(f"\n  By category:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<14} {n:>7,}")

    print(f"\n  By extension:")
    for ext, n in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext:<14} {n:>7,}")

    # Top 10 most-chunked files
    top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 most-chunked files:")
    for fp, n in top_files:
        short = fp[-70:] if len(fp) > 70 else fp
        print(f"    {n:>4}x  {short}")
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Render a single chunk
# ---------------------------------------------------------------------------

def render_chunk(idx: int, total: int, chunk: dict) -> None:
    file_path  = chunk.get("file_path", "unknown")
    ext        = chunk.get("ext", "")
    category   = chunk.get("category", "")
    chunk_i    = chunk.get("chunk_index", 0)
    total_c    = chunk.get("total_chunks", 1)
    start_char = chunk.get("start_char", 0)
    end_char   = chunk.get("end_char", 0)
    start_page = chunk.get("start_page", None)
    has_embed  = "embedding" in chunk

    text = chunk.get("text", "")

    # Split header / body
    if "body:\n" in text:
        header_raw = text.split("body:\n", 1)[0]
        body_raw   = text.split("body:\n", 1)[1]
        header_raw = header_raw.split("header:\n", 1)[1] if "header:\n" in header_raw else header_raw
    elif "header:\n" in text:
        header_raw = text.split("header:\n", 1)[1]
        body_raw   = ""
    else:
        header_raw = ""
        body_raw   = text

    header_preview = header_raw.strip()[:300].replace("\n", " ↵ ")
    body_preview   = body_raw.strip()[:PREVIEW_CHARS]

    sep = "─" * WRAP_WIDTH
    print(f"\n{sep}")
    print(f"  [{idx+1}/{total}]  {file_path}")
    print(f"  ext={ext}  cat={category}  chunk {chunk_i+1}/{total_c}  "
          f"chars {start_char:,}–{end_char:,}"
          + (f"  page≈{start_page}" if start_page else "")
          + (f"  [embedded]" if has_embed else "  [NO EMBEDDING]"))
    print(sep)

    if header_preview:
        print(f"\n  HEADER PREVIEW:\n  {header_preview}\n")

    print("  BODY:")
    for line in body_preview.splitlines():
        print("  " + line)

    body_chars = len(body_raw.strip())
    if body_chars == 0:
        print("\n  ⚠️  EMPTY BODY — this chunk has no body text!")
    elif body_chars < 200:
        print(f"\n  ⚠️  SHORT BODY ({body_chars} chars)")

    if len(body_raw.strip()) > PREVIEW_CHARS:
        print(f"\n  ... [{len(body_raw.strip()) - PREVIEW_CHARS:,} more chars not shown]")
    print()

# ---------------------------------------------------------------------------
# Interactive browse
# ---------------------------------------------------------------------------

def browse(chunks: list[dict]) -> None:
    idx   = 0
    total = len(chunks)
    if total == 0:
        print("No chunks to browse.")
        return

    print(f"\nBrowsing {total:,} chunks. Commands: ENTER=next  b=back  s=skip10  f=jump  q=quit\n")

    while 0 <= idx < total:
        render_chunk(idx, total, chunks[idx])
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "b":
            idx = max(0, idx - 1)
        elif cmd == "s":
            idx = min(total - 1, idx + 10)
        elif cmd == "f":
            try:
                target = int(input("  Jump to index (1-based): ")) - 1
                idx = max(0, min(total - 1, target))
            except ValueError:
                print("  Invalid number.")
        else:
            idx += 1  # ENTER or anything else = next

    print("\nDone browsing.")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_file(chunks: list[dict], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            file_path = chunk.get("file_path", "unknown")
            text      = chunk.get("text", "")
            body      = text.split("body:\n", 1)[1] if "body:\n" in text else text
            f.write(f"=== CHUNK {i+1}  {file_path}  [chunk {chunk.get('chunk_index',0)+1}/{chunk.get('total_chunks',1)}] ===\n")
            f.write(body.strip() + "\n\n")
    print(f"Exported {len(chunks):,} chunks -> {out_path.resolve()}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Review chunks.jsonl quality.")
    parser.add_argument("--file",   default=None,  help="Path to JSONL file")
    parser.add_argument("--ext",    default=None,  help="Filter by extension, e.g. .pdf")
    parser.add_argument("--search", default=None,  help="Keyword filter on chunk text")
    parser.add_argument("--limit",  type=int, default=None, help="Max chunks to load")
    parser.add_argument("--stats",  action="store_true",    help="Print stats only, skip browser")
    parser.add_argument("--export", default=None,  help="Export all matching chunks to a text file")
    args = parser.parse_args()

    # Resolve input file
    if args.file:
        input_path = Path(args.file)
    else:
        for candidate in ("chunks.jsonl", "chunks_embedded.jsonl"):
            if Path(candidate).exists():
                input_path = Path(candidate)
                break
        else:
            print("ERROR: No chunks.jsonl or chunks_embedded.jsonl found. Pass --file explicitly.")
            return

    print(f"Loading {input_path} ...")
    chunks = load_chunks(input_path, args.ext, args.search)

    filters = []
    if args.ext:    filters.append(f"ext={args.ext}")
    if args.search: filters.append(f"search='{args.search}'")
    if filters:
        print(f"  Filtered to {len(chunks):,} chunks ({', '.join(filters)})")

    if args.limit:
        chunks = chunks[:args.limit]
        print(f"  Capped at {len(chunks):,} chunks (--limit {args.limit})")

    print_stats(chunks)

    if args.export:
        export_to_file(chunks, Path(args.export))
        return

    if not args.stats:
        browse(chunks)


if __name__ == "__main__":
    main()