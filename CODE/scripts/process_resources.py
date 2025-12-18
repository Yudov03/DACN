"""
Process Resources Script
=========================

Scan data/resource/ folder and PROCESS documents (OCR/ASR) WITHOUT indexing.
Saves RAW processed content to data/knowledge_base/processed/*.json

This is the "pre-processing" step - useful for:
- Batch OCR/ASR processing (takes long time)
- Testing document processing pipeline
- Preparing data for later indexing with reindex_documents.py

Pipeline: Scan → Process (OCR/ASR) → Save RAW JSON (NO post-processing)

Post-processing is done in reindex_documents.py, NOT here.
This matches the flow in KnowledgeBase.add_document():
    1. Process → 2. Save RAW → 3. Post-process → 4. Chunk → 5. Index

Usage:
    python scripts/process_resources.py                # Process all
    python scripts/process_resources.py --dry-run      # Preview only
    python scripts/process_resources.py --clear        # Clear KB first (same as import_resources)
    python scripts/process_resources.py --skip-existing  # Skip already processed files

Workflow comparison:
    process_resources.py  → Produces: processed/*.json (RAW, no post-process)
    reindex_documents.py  → Reads: processed/*.json → Post-process → Chunk → Index
    import_resources.py   → Does ALL in one step
"""

import os
import sys
import io
import gc
import json
import time
import hashlib
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
    MEDIA_TYPES,
)


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
    print(f"    [Memory: {mem_mb:.0f}MB | System: {mem_percent:.0f}% | {status}]", flush=True)


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def scan_resources(resource_dir: Path):
    """Scan resource folder for supported files"""
    files = []

    for file_path in resource_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
            files.append(file_path)

    return sorted(files)


def load_existing_registry(index_path: Path) -> dict:
    """Load existing index.json registry"""
    if index_path.exists():
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("documents", {})
        except Exception as e:
            log(f"Error loading registry: {e}", "WARN")
    return {}


def find_doc_by_hash(registry: dict, file_hash: str) -> str:
    """Find document ID by file hash"""
    for doc_id, doc_info in registry.items():
        if doc_info.get("file_hash") == file_hash:
            return doc_id
    return None


def process_resources(
    dry_run: bool = False,
    clear_first: bool = False,
    skip_existing: bool = False
):
    """Process all resources from data/resource/ and save RAW content to processed/*.json"""

    from modules.document_processor import UnifiedProcessor
    from modules.file_types import get_folder_for_type

    resource_dir = PROJECT_ROOT / "data" / "resource"
    kb_dir = PROJECT_ROOT / "data" / "knowledge_base"
    processed_dir = kb_dir / "processed"
    documents_dir = kb_dir / "documents"
    transcripts_dir = kb_dir / "transcripts"
    index_path = kb_dir / "index.json"

    print("=" * 60)
    print("  PROCESS RESOURCES (NO INDEXING)")
    print("=" * 60)
    start_time = time.time()

    # Create directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    # Clear everything if requested (same as KnowledgeBase.clear_all())
    if clear_first and not dry_run:
        log("Clearing Knowledge Base...", "WAIT")

        # Clear files in documents/, processed/, transcripts/ (keep folder structure)
        cleared_files = 0
        for folder in [documents_dir, processed_dir, transcripts_dir]:
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        try:
                            f.unlink()
                            cleared_files += 1
                        except Exception as e:
                            log(f"Error deleting {f}: {e}", "WARN")

        log(f"Cleared {cleared_files} files (kept folder structure)", "OK")

        # Reset index.json
        if index_path.exists():
            index_path.unlink()
        log("Cleared index.json", "OK")

        # Note: Qdrant will be cleared when running reindex_documents.py
        log("Note: Run reindex_documents.py to clear Qdrant", "INFO")

        # Reset registry
        registry = {}
    else:
        # Load existing registry
        registry = load_existing_registry(index_path)
        log(f"Existing registry: {len(registry)} documents", "INFO")

    # Scan files
    log(f"Scanning: {resource_dir}", "WAIT")
    files = scan_resources(resource_dir)

    if not files:
        log("No supported files found in data/resource/", "ERROR")
        log("Please add documents/audio to:", "INFO")
        print(f"  - {resource_dir / 'documents'}")
        print(f"  - {resource_dir / 'audio'}")
        return

    # Categorize files
    docs = [f for f in files if f.suffix.lower() in DOCUMENT_EXTENSIONS]
    audios = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS]

    log(f"Found {len(files)} files:", "OK")
    print(f"  - Documents: {len(docs)}")
    print(f"  - Audio: {len(audios)}")
    print(f"  - Video: {len(videos)}")

    # Order: Documents first (lighter), then Media (heavier)
    ordered_files = docs + videos + audios

    # Check for existing files if skip_existing
    files_to_process = []
    skipped_count = 0

    for f in ordered_files:
        file_hash = compute_file_hash(f)
        existing_doc_id = find_doc_by_hash(registry, file_hash)

        if skip_existing and existing_doc_id:
            # Check if processed file exists
            processed_file = processed_dir / f"{existing_doc_id}.json"
            if processed_file.exists():
                skipped_count += 1
                continue

        files_to_process.append((f, file_hash, existing_doc_id))

    if skip_existing and skipped_count > 0:
        log(f"Skipping {skipped_count} already processed files", "INFO")

    if not files_to_process:
        log("No new files to process!", "OK")
        return

    # List files
    print("\n--- Files to process ---")
    for i, (f, _, _) in enumerate(files_to_process, 1):
        rel_path = f.relative_to(resource_dir)
        size_kb = f.stat().st_size / 1024
        print(f"  [{i:2}] {rel_path} ({size_kb:.1f} KB)")

    if dry_run:
        log("DRY RUN - No changes made.", "WARN")
        return

    # Initialize processor
    log("Initializing UnifiedProcessor...", "WAIT")
    processor = UnifiedProcessor()
    log("UnifiedProcessor ready", "OK")
    log("Note: Post-processing will be done in reindex_documents.py", "INFO")

    # Process files
    print("\n--- Processing Files ---")
    success = 0
    failed = 0
    import uuid

    for i, (file_path, file_hash, existing_doc_id) in enumerate(files_to_process, 1):
        rel_path = file_path.relative_to(resource_dir)
        print(f"\n{'='*60}")
        log(f"[{i}/{len(files_to_process)}] {rel_path}", "WAIT")

        # Generate or reuse doc_id
        if existing_doc_id:
            doc_id = existing_doc_id
            log(f"  Reusing doc_id: {doc_id}", "INFO")
        else:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            log(f"  New doc_id: {doc_id}", "INFO")

        file_type = file_path.suffix.lower().strip('.')
        log(f"  Type: {file_type}", "INFO")

        try:
            # 1. Process document (OCR/ASR)
            log("  Processing (OCR/ASR if needed)...", "WAIT")
            process_start = time.time()

            processed = processor.process(str(file_path))

            process_time = time.time() - process_start

            if not processed.success:
                raise Exception(processed.error_message or "Processing failed")

            log(f"  Processed: {len(processed.content):,} chars ({process_time:.1f}s)", "OK")
            log(f"  Extraction type: {processed.extraction_type}", "INFO")

            # 2. Save RAW processed result to JSON (NO post-processing here)
            processed_path = processed_dir / f"{doc_id}.json"

            processed_data = {
                "content": processed.content,  # RAW content, post-process later
                "file_type": file_type,
                "extraction_type": processed.extraction_type,
                "metadata": processed.metadata.to_dict() if hasattr(processed.metadata, 'to_dict') else {},
                "tables_count": len(processed.tables) if processed.tables else 0,
                "processed_at": datetime.now().isoformat(),
                "processing_time": process_time,
            }

            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            log(f"  Saved: {processed_path.name}", "OK")

            # 3. Save transcript for audio/video
            if file_type in MEDIA_TYPES and processed.content:
                transcripts_dir.mkdir(parents=True, exist_ok=True)
                transcript_path = transcripts_dir / f"{doc_id}.txt"
                safe_filename = file_path.name.encode('utf-8', errors='replace').decode('utf-8')
                with open(transcript_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"# Transcript: {safe_filename}\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                    f.write(processed.content)
                log(f"  Transcript saved: {transcript_path.name}", "OK")

            # 4. Copy file to KB documents/
            folder = get_folder_for_type(file_type)
            dest_dir = documents_dir / folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{doc_id}{file_path.suffix}"

            if not dest_path.exists():
                import shutil
                shutil.copy2(file_path, dest_path)
                log(f"  Copied to: documents/{folder}/{dest_path.name}", "OK")
            else:
                log(f"  File exists: documents/{folder}/{dest_path.name}", "INFO")

            # 5. Update registry
            registry[doc_id] = {
                "id": doc_id,
                "filename": file_path.name,
                "original_path": f"documents/{folder}/{dest_path.name}",
                "processed_path": f"processed/{doc_id}.json",
                "file_type": file_type,
                "file_size": file_path.stat().st_size,
                "file_hash": file_hash,
                "page_count": processed.metadata.page_count if processed.metadata else None,
                "chunk_count": 0,  # Not chunked yet
                "chunk_ids": [],   # Not indexed yet
                "metadata": {},
                "tags": [],
                "category": None,
                "uploaded_at": datetime.now().isoformat(),
                "processed_at": datetime.now().isoformat(),
                "updated_at": None,
                "status": "processed",  # Not "indexed" yet!
                "error_message": None,
            }

            print_memory_status()
            success += 1

        except Exception as e:
            log(f"  FAILED: {e}", "ERROR")
            print_memory_status()
            failed += 1

        # Force garbage collection
        gc.collect()

        # Check memory
        _, mem_percent = get_memory_usage()
        if mem_percent > 90:
            log("  Memory critical! Forcing cleanup...", "WARN")
            gc.collect()
            gc.collect()

    # Save updated index.json
    print("\n--- Saving Index ---")
    index_data = {
        "version": "1.0",
        "updated_at": datetime.now().isoformat(),
        "documents": registry,
        "stats": {
            "total_documents": len(registry),
            "total_chunks": sum(d.get("chunk_count", 0) for d in registry.values()),
        }
    }

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    log(f"Index saved: {index_path}", "OK")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETED")
    print("=" * 60)
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", "OK")
    log(f"Success: {success}", "OK")
    log(f"Failed: {failed}", "ERROR" if failed > 0 else "OK")
    log(f"Total documents in registry: {len(registry)}", "INFO")

    print("\n--- NEXT STEP ---")
    log("To index these documents to Qdrant, run:", "INFO")
    print("  python scripts/reindex_documents.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Process resources WITHOUT indexing (OCR/ASR only, saves RAW content)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without processing"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear Knowledge Base (processed/, documents/, transcripts/, index.json) before processing"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that are already processed"
    )

    args = parser.parse_args()

    process_resources(
        dry_run=args.dry_run,
        clear_first=args.clear,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
