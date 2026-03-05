#!/usr/bin/env python3
"""
scan_school_dir.py — Read-only directory scanner.
Nothing is moved, modified, or deleted.
Only counts files in the approved extension allowlist.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path("../../School")

# Only show/save extensions with at least this many files. Set to 1 to show all.
MIN_FILE_COUNT = 1

# Only these extensions will be counted/kept. Everything else is ignored.
ALLOWLIST = {
    # Plaintext / code
    ".md", ".mdx", ".py", ".cpp", ".c", ".h", ".html", ".js", ".jsx",
    ".cjs", ".mjs", ".css", ".sql", ".hs", ".r", ".bash", ".yaml", ".yml",
    ".toml", ".cfg", ".xml", ".msg", ".srv", ".urdf", ".txt", ".csv",
    ".json", ".ipynb", ".asm", ".sv", ".sh", ".tsl", ".plain", ".cmake", ".in",
    # Documents
    ".pdf", ".docx", ".pptx", ".xlsx",
    # Images
    ".jpg", ".jpeg", ".png", ".webp",
}

# Directory names to skip entirely (os.walk won't descend into them)
SKIP_DIRS = {
    ".git", ".idea", ".vscode",
    "__pycache__", "venv", ".venv", "env", "build", "dist", "target",
    ".gradle", ".next", ".nuxt", "out", "coverage",
    ".cache", ".pytest_cache", ".mypy_cache", "eggs", ".eggs",
    "node_modules",
    ".vscode-server", ".nvm", ".dotnet", ".forever", ".local", ".ssh", ".npm",
    "sample-data",
}

# Skip any file whose full path contains one of these substrings (case-insensitive)
SKIP_PATH_FRAGMENTS = {
    "site-packages",
    "dist-info",
    "scikit-learn",
    "pyarrow",
    "huggingface",
    "palm/media",
    "palm/tests/artifacts",
    "palm/lerobot",
    "cs_361/archive",
}

# Skip these exact filenames anywhere in the tree
SKIP_FILENAMES = {
    "package-lock.json",
    "CMakeLists.txt",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS

def should_skip_file(fpath: Path) -> bool:
    if fpath.name in SKIP_FILENAMES:
        return True
    path_str = str(fpath).replace("\\", "/").lower()
    return any(frag.lower() in path_str for frag in SKIP_PATH_FRAGMENTS)

def human_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan(root: Path):
    stats = defaultdict(lambda: {"count": 0, "total_bytes": 0, "examples": []})
    total_files = skipped_dirs = skipped_files = ignored_ext = 0

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        before = len(dirnames)
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        skipped_dirs += before - len(dirnames)

        for fname in filenames:
            fpath = current / fname
            ext = fpath.suffix.lower() if fpath.suffix else "(no extension)"

            # Extension not in allowlist — skip before any other check
            if ext not in ALLOWLIST:
                ignored_ext += 1
                continue

            if should_skip_file(fpath):
                skipped_files += 1
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                size = 0

            s = stats[ext]
            s["count"] += 1
            s["total_bytes"] += size
            if len(s["examples"]) < 3:
                s["examples"].append(str(fpath.relative_to(root)))

            total_files += 1

    return dict(stats), total_files, skipped_dirs, skipped_files, ignored_ext

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_report(stats, total_files, skipped_dirs, skipped_files, ignored_ext, root):
    sorted_stats = sorted(stats.items(), key=lambda x: -x[1]["count"])
    col_w = [22, 10, 12, 55]
    divider = "-" * sum(col_w)

    print(f"\n  Scanning: {root.resolve()}")
    print(f"{'=' * sum(col_w)}")
    print(f"{'Extension':<{col_w[0]}} {'Count':>{col_w[1]}} {'Total Size':>{col_w[2]}}  Example Path")
    print(divider)

    for ext, data in sorted_stats:
        ex = data["examples"][0] if data["examples"] else ""
        if len(ex) > col_w[3]:
            ex = "..." + ex[-(col_w[3]-1):]
        print(f"{ext:<{col_w[0]}} {data['count']:>{col_w[1]},} {human_size(data['total_bytes']):>{col_w[2]}}  {ex}")

    print(divider)
    print(f"\n  Kept files                  : {total_files:,}")
    print(f"  Dirs skipped (junk)         : {skipped_dirs:,}")
    print(f"  Files skipped (path filter) : {skipped_files:,}")
    print(f"  Files ignored (wrong ext)   : {ignored_ext:,}")

    categories = {
        "Documents": {".pdf",".docx",".doc",".pptx",".xlsx",".rtf"},
        "Code":      {".py",".cpp",".c",".h",".js",".jsx",".cjs",".mjs",".ts",
                      ".html",".css",".sql",".hs",".r",".sh",".bash",".asm",".sv"},
        "Notes":     {".md",".mdx",".txt",".plain",".ipynb"},
        "Config":    {".json",".yaml",".yml",".toml",".cfg",".xml",".cmake",
                      ".in",".msg",".srv",".urdf",".tsl"},
        "Data":      {".csv"},
        "Images":    {".jpg",".jpeg",".png",".webp"},
    }

    print(f"\n  Category Breakdown\n  {'-'*50}")
    accounted = set()
    for name, exts in categories.items():
        c = sum(stats[e]["count"] for e in exts if e in stats)
        b = sum(stats[e]["total_bytes"] for e in exts if e in stats)
        if c:
            print(f"  {name:<22} {c:>6,} files   {human_size(b):>10}")
        accounted |= exts
    other_exts = set(stats) - accounted
    oc = sum(stats[e]["count"] for e in other_exts)
    ob = sum(stats[e]["total_bytes"] for e in other_exts)
    if oc:
        print(f"  {'Other':<22} {oc:>6,} files   {human_size(ob):>10}")
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1])
    if not ROOT.exists():
        print(f"Directory not found: {ROOT}")
        sys.exit(1)

    print("Scanning...")
    stats, total_files, skipped_dirs, skipped_files, ignored_ext = scan(ROOT)
    stats = {k: v for k, v in stats.items() if v["count"] >= MIN_FILE_COUNT}
    print_report(stats, total_files, skipped_dirs, skipped_files, ignored_ext, ROOT)

    out = Path("school_scan_results.json")
    with open(out, "w") as f:
        json.dump({
            "root": str(ROOT.resolve()),
            "total_files": total_files,
            "skipped_files": skipped_files,
            "ignored_ext": ignored_ext,
            "extensions": stats,
        }, f, indent=2)
    print(f"  Saved -> {out.resolve()}\n")