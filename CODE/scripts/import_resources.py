"""
Import Resources Script
========================

Scan data/resource/ folder and import all documents/audio to Knowledge Base.
This is the ALL-IN-ONE script that does: Process → Post-process → Chunk → Embed → Index

Usage:
    python scripts/import_resources.py              # Import all
    python scripts/import_resources.py --dry-run    # Preview only
    python scripts/import_resources.py --clear      # Clear KB first

Alternative 2-step workflow:
    python scripts/process_resources.py             # Step 1: OCR/ASR only
    python scripts/reindex_documents.py             # Step 2: Post-process + Index

This script is used by ADMIN (school) to load all resources into the system.
"""

import os
import sys
import io
import gc
import time
import argparse
import psutil
from pathlib import Path
from datetime import datetime

# Disable PaddleOCR model source check (speeds up startup)
os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("[STARTUP] Script starting...", flush=True)


def log(msg: str, level: str = "INFO"):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "i", "OK": "+", "WARN": "!", "ERROR": "x", "WAIT": "..."}.get(level, "*")
    print(f"[{timestamp}] {prefix} {msg}", flush=True)


def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    mem_percent = psutil.virtual_memory().percent
    return mem_mb, mem_percent


def print_memory_status():
    """Print memory status"""
    mem_mb, mem_percent = get_memory_usage()
    status = "OK" if mem_percent < 80 else "WARNING" if mem_percent < 90 else "CRITICAL"
    print(f"    [Memory: {mem_mb:.0f}MB | System: {mem_percent:.0f}% | {status}]")

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized file types - SINGLE SOURCE OF TRUTH
from modules.file_types import (
    ALL_SUPPORTED_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
)


def scan_resources(resource_dir: Path):
    """Scan resource folder for supported files"""
    files = []

    for file_path in resource_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
            files.append(file_path)

    return sorted(files)


def import_all_resources(dry_run: bool = False, clear_first: bool = False):
    """Import all resources from data/resource/ to Knowledge Base"""

    from modules import KnowledgeBase

    resource_dir = PROJECT_ROOT / "data" / "resource"
    kb_dir = PROJECT_ROOT / "data" / "knowledge_base"

    print("=" * 60)
    print("  IMPORT RESOURCES TO KNOWLEDGE BASE")
    print("  (Process → Post-process → Chunk → Embed → Index)")
    print("=" * 60)
    start_time = time.time()

    # Scan files
    log(f"Scanning: {resource_dir}", "WAIT")
    files = scan_resources(resource_dir)

    if not files:
        log("No supported files found in data/resource/", "ERROR")
        log("Please add documents/audio to:", "INFO")
        print(f"  - {resource_dir / 'documents'}")
        print(f"  - {resource_dir / 'audio'}")
        return

    # Categorize files using centralized extension sets (from file_types module)
    docs = [f for f in files if f.suffix.lower() in DOCUMENT_EXTENSIONS]
    audios = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS]

    log(f"Found {len(files)} files:", "OK")
    print(f"  - Documents: {len(docs)}")
    print(f"  - Audio: {len(audios)}")
    print(f"  - Video: {len(videos)}")

    # Reorder files: Documents first (lighter), then Audio/Video (heavier)
    # This ensures PaddleOCR loads when memory is still available
    ordered_files = docs + videos + audios

    # List files
    print("\n--- Files to import (Documents first, then Media) ---")
    for i, f in enumerate(ordered_files, 1):
        rel_path = f.relative_to(resource_dir)
        size_kb = f.stat().st_size / 1024
        print(f"  [{i:2}] {rel_path} ({size_kb:.1f} KB)")

    files = ordered_files  # Use ordered list

    if dry_run:
        log("DRY RUN - No changes made.", "WARN")
        return

    # Initialize Knowledge Base
    print("\n--- Initializing Knowledge Base ---")
    log("Loading KnowledgeBase...", "WAIT")
    kb = KnowledgeBase(base_dir=str(kb_dir))
    log("KnowledgeBase ready", "OK")

    # Clear if requested
    if clear_first:
        log("Clearing existing Knowledge Base...", "WAIT")
        result = kb.clear_all()
        log(f"Cleared Qdrant: {result['chunks']} points", "OK")
        log(f"Cleared KB: {result['documents']} documents", "OK")
        log(f"Cleared files: {result['files']} files", "OK")

    # Import files
    print("\n--- Importing Files ---")
    success = 0
    failed = 0

    for i, file_path in enumerate(files, 1):
        rel_path = file_path.relative_to(resource_dir)
        file_start = time.time()
        print(f"\n{'='*60}")
        log(f"[{i}/{len(files)}] {rel_path}", "WAIT")

        # Determine tags based on folder and file type
        tags = []
        rel_path_str = str(rel_path).lower()

        # Folder-based tags
        if "documents" in rel_path_str:
            tags.append("document")
        if "audio" in rel_path_str:
            tags.append("audio")
        if "video" in rel_path_str:
            tags.append("video")

        # File type based tag (if not already tagged by folder)
        ext = file_path.suffix.lower()
        if ext in AUDIO_EXTENSIONS and "audio" not in tags:
            tags.append("audio")
        elif ext in VIDEO_EXTENSIONS and "video" not in tags:
            tags.append("video")
        elif ext in DOCUMENT_EXTENSIONS and "document" not in tags:
            tags.append("document")

        # Add file extension as tag
        tags.append(ext.replace('.', ''))

        try:
            log(f"  Processing (OCR/ASR → Post-process → Chunk → Embed → Index)...", "WAIT")
            doc_id = kb.add_document(str(file_path), tags=tags)
            file_time = time.time() - file_start
            log(f"  Imported: {doc_id} ({file_time:.1f}s)", "OK")
            print_memory_status()
            success += 1
        except Exception as e:
            file_time = time.time() - file_start
            log(f"  FAILED: {e} ({file_time:.1f}s)", "ERROR")
            print_memory_status()
            failed += 1

        # Force garbage collection after each file to free memory
        gc.collect()

        # Check memory - warn if critical
        _, mem_percent = get_memory_usage()
        if mem_percent > 90:
            log("  Memory critical! Forcing cleanup...", "WARN")
            gc.collect()
            gc.collect()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  IMPORT COMPLETED")
    print("=" * 60)

    stats = kb.get_stats()
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", "OK")
    log(f"Success: {success}", "OK")
    log(f"Failed: {failed}", "ERROR" if failed > 0 else "OK")
    print(f"\nKnowledge Base:")
    log(f"Total documents: {stats.total_documents}", "INFO")
    log(f"Total chunks: {stats.total_chunks}", "INFO")
    log(f"Total size: {stats.total_size_bytes / 1024 / 1024:.2f} MB", "INFO")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Import all resources to Knowledge Base"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without importing"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear Knowledge Base before importing"
    )

    args = parser.parse_args()

    import_all_resources(
        dry_run=args.dry_run,
        clear_first=args.clear
    )


if __name__ == "__main__":
    main()
