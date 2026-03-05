#!/usr/bin/env python3
"""
build_ocr_list.py — Writes ocr_candidates.txt: one PDF per line that has at
least one image-based page needing OCR, minus PDFs that are likely exports of
another file (same stem, different extension in the same folder).

Edit ocr_candidates.txt to remove any files you don't want, then run:
    python ocr_pdf_pages.py --list ocr_candidates.txt
"""

import re
import os
from pathlib import Path
from collections import defaultdict

SCHOOL_ROOT = "../../School"
LOG_FILE    = "ingest_errors.log"
OUTPUT_FILE = "ocr_candidates.txt"

SKIP_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", "venv", ".venv", "env",
    "build", "dist", "target", ".gradle", ".next", ".nuxt", "out",
    "coverage", ".cache", ".pytest_cache", ".mypy_cache", "eggs", ".eggs",
    "node_modules", ".vscode-server", ".nvm", ".dotnet", ".forever",
    ".local", ".ssh", ".npm", "sample-data",
}
SKIP_PATH_FRAGMENTS = {
    "site-packages", "dist-info", "scikit-learn", "pyarrow",
    "huggingface", "palm/media", "palm/tests/artifacts", "palm/lerobot",
    "cs_361/archive",
}

def parse_error_log(log_path):
    """Returns set of PDF rel_paths that have any skipped pages."""
    files = set()
    pattern = re.compile(r"^SKIP_(?:PAGE|FILE) (.+?)(?:\s+page \d+)?:")
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                files.add(m.group(1))
    return files

def find_duplicate_stems(school_root):
    """Returns set of PDF rel_paths that share a stem with another file in the same folder."""
    dupes = set()
    for dirpath, dirnames, filenames in os.walk(school_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        current = Path(dirpath)
        path_str = str(current).replace("\\", "/").lower()
        if any(frag in path_str for frag in SKIP_PATH_FRAGMENTS):
            continue
        stem_map = defaultdict(list)
        for fname in filenames:
            fpath = current / fname
            stem_map[fpath.stem.lower()].append(fpath.suffix.lower())
        for stem, exts in stem_map.items():
            if ".pdf" in exts and len(exts) > 1:
                for fname in filenames:
                    fpath = current / fname
                    if fpath.suffix.lower() == ".pdf" and fpath.stem.lower() == stem:
                        try:
                            rel = str(fpath.relative_to(school_root)).replace("\\", "/")
                        except ValueError:
                            rel = str(fpath)
                        dupes.add(rel)
    return dupes

def main():
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print(f"ERROR: {LOG_FILE} not found. Run ingest.py first.")
        return

    candidates = parse_error_log(log_path)
    dupes      = find_duplicate_stems(Path(SCHOOL_ROOT))

    before = len(candidates)
    candidates -= dupes
    removed = before - len(candidates)

    sorted_files = sorted(candidates)
    Path(OUTPUT_FILE).write_text("\n".join(sorted_files) + "\n", encoding="utf-8")

    print(f"Candidates from log : {before}")
    print(f"Removed (dupe stem) : {removed}")
    print(f"Written             : {len(sorted_files)} -> {OUTPUT_FILE}")
    print("Delete any lines you don't want OCR'd, then run:")
    print("  python ocr_pdf_pages.py --list ocr_candidates.txt")

if __name__ == "__main__":
    main()