"""
Re-index Processed Documents
============================

Read already processed documents and re-index them to Qdrant.
This script does NOT process files - it uses existing processed/*.json files.

Workflow:
    1. Read processed/*.json files
    2. Post-process content (fix OCR/ASR errors)
    3. Chunk text
    4. Upload to Qdrant (with reset)

Usage:
    python scripts/reindex_documents.py                        # Re-index all
    python scripts/reindex_documents.py --dry-run              # Preview only
    python scripts/reindex_documents.py --no-reset             # Don't reset Qdrant
    python scripts/reindex_documents.py --method none          # Skip post-processing
    python scripts/reindex_documents.py --method transformer   # Use transformer (fast)
    python scripts/reindex_documents.py --file doc_8786094a    # Process single file
    python scripts/reindex_documents.py --sentence-mode        # 1 sentence = 1 chunk (slower but more accurate)
"""

import os
import sys
import io
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("[STARTUP] Script starting...", flush=True)
print(f"[STARTUP] Python: {sys.version}", flush=True)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

print(f"[STARTUP] Project root: {PROJECT_ROOT}", flush=True)


def log(msg: str, level: str = "INFO"):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "i", "OK": "+", "WARN": "!", "ERROR": "x", "WAIT": "..."}.get(level, "*")
    print(f"[{timestamp}] {prefix} {msg}", flush=True)


def check_ollama_status(base_url: str = "http://localhost:11434") -> dict:
    """Check Ollama server status and loaded models."""
    status = {
        "server_running": False,
        "server_url": base_url,
        "models_available": [],
        "error": None
    }

    try:
        # Check if server is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            status["server_running"] = True
            data = response.json()
            status["models_available"] = [m["name"] for m in data.get("models", [])]
        else:
            status["error"] = f"HTTP {response.status_code}"
    except requests.ConnectionError:
        status["error"] = "Cannot connect to Ollama server"
    except requests.Timeout:
        status["error"] = "Connection timeout"
    except Exception as e:
        status["error"] = str(e)

    return status


def load_processed_documents(processed_dir: Path) -> list:
    """Load all processed JSON files."""
    documents = []

    for json_file in processed_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file_path'] = json_file
                data['_doc_id'] = json_file.stem  # e.g., "doc_8786094a"
                documents.append(data)
        except Exception as e:
            print(f"  [ERROR] Failed to load {json_file.name}: {e}")

    return documents


def determine_extraction_type(file_type: str) -> str:
    """Determine if extraction was direct or indirect."""
    file_type_lower = file_type.lower()

    # Indirect: OCR (pdf, images) or ASR (audio, video)
    indirect_extensions = {
        'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp',  # OCR
        'mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma', 'aac',  # Audio ASR
        'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'm4v'  # Video ASR
    }

    # Also check for generic types (stored in processed JSON)
    indirect_types = {'audio', 'video', 'image'}

    if file_type_lower in indirect_extensions or file_type_lower in indirect_types:
        return "indirect"
    return "direct"


def reindex_documents(
    dry_run: bool = False,
    reset_qdrant: bool = True,
    postprocess_method: str = None,
    single_file: str = None,
    sentence_mode: bool = False
):
    """Re-index all processed documents to Qdrant."""

    from src.modules import TextChunker, TextEmbedding, VectorDatabase

    kb_dir = PROJECT_ROOT / "data" / "knowledge_base"
    processed_dir = kb_dir / "processed"
    index_path = kb_dir / "index.json"

    print("=" * 60)
    if single_file:
        print(f"  RE-INDEX SINGLE FILE: {single_file}")
    else:
        print("  RE-INDEX PROCESSED DOCUMENTS")
    print("=" * 60)
    start_time = time.time()

    # ===== STEP 1: Check Ollama status =====
    print("\n--- Step 1: Check Ollama Status ---")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("POSTPROCESS_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))

    ollama_status = check_ollama_status(ollama_url)

    if ollama_status["server_running"]:
        log(f"Ollama server: RUNNING at {ollama_url}", "OK")
        log(f"Available models: {', '.join(ollama_status['models_available'])}", "INFO")

        if ollama_model in ollama_status["models_available"]:
            log(f"Target model '{ollama_model}': AVAILABLE", "OK")
        else:
            log(f"Target model '{ollama_model}': NOT FOUND (will download or fail)", "WARN")
    else:
        log(f"Ollama server: NOT RUNNING - {ollama_status['error']}", "ERROR")
        log("Post-processing with Ollama will be skipped or fail", "WARN")

    # ===== STEP 2: Check processed directory =====
    print("\n--- Step 2: Load Processed Documents ---")
    if not processed_dir.exists():
        log(f"Processed directory not found: {processed_dir}", "ERROR")
        return

    # Handle single file mode
    if single_file:
        # Normalize filename (add .json if needed)
        if not single_file.endswith('.json'):
            single_file = single_file + '.json'

        target_file = processed_dir / single_file
        if not target_file.exists():
            log(f"File not found: {target_file}", "ERROR")
            log(f"Available files in {processed_dir}:", "INFO")
            for f in processed_dir.glob("*.json"):
                print(f"  - {f.name}")
            return

        log(f"Loading single file: {single_file}", "INFO")
        documents = load_processed_documents(processed_dir)
        # Filter to only the specified file
        documents = [d for d in documents if d['_file_path'].name == single_file]
    else:
        log(f"Loading from: {processed_dir}", "INFO")
        documents = load_processed_documents(processed_dir)

    if not documents:
        log("No processed documents found!", "ERROR")
        return

    log(f"Found {len(documents)} document(s) to process", "OK")

    # Load index.json for metadata
    doc_registry = {}
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
            doc_registry = index_data.get("documents", {})

    # Count by extraction type
    # Use extraction_type from JSON (new format) or fallback to determine_extraction_type (old format)
    direct_count = 0
    indirect_count = 0
    for doc in documents:
        ext_type = doc.get('extraction_type') or determine_extraction_type(doc.get('file_type', ''))
        if ext_type == "direct":
            direct_count += 1
        else:
            indirect_count += 1

    log(f"Direct extraction (txt, docx...): {direct_count} files", "INFO")
    log(f"Indirect extraction (OCR/ASR): {indirect_count} files", "INFO")

    # Show documents
    print("\n--- Documents to re-index ---")
    for i, doc in enumerate(documents, 1):
        doc_id = doc['_doc_id']
        file_type = doc.get('file_type', 'unknown')
        content_len = len(doc.get('content', ''))
        # Use extraction_type from JSON (new format) or fallback to determine_extraction_type (old format)
        extraction_type = doc.get('extraction_type') or determine_extraction_type(file_type)

        # Get filename from registry
        filename = "unknown"
        if doc_id in doc_registry:
            filename = doc_registry[doc_id].get('filename', 'unknown')

        print(f"  [{i}] {filename}")
        print(f"      ID: {doc_id} | Type: {file_type} | {extraction_type} | {content_len:,} chars")

    if dry_run:
        log("DRY RUN - No changes made.", "WARN")
        return

    # ===== STEP 3: Initialize components =====
    print("\n--- Step 3: Initialize Components ---")

    from src.modules.post_processing import get_processor
    log("Loading PostProcessor...", "WAIT")
    processor = get_processor()  # Use singleton
    proc_info = processor.get_info()
    log(f"PostProcessor ready", "OK")
    log(f"  Direct method: {proc_info['direct_method']}", "INFO")
    log(f"  Indirect method: {proc_info['indirect_method']}", "INFO")
    log(f"  Ollama model: {proc_info['ollama_model']}", "INFO")
    log(f"  Transformer model: {proc_info['transformer_model']}", "INFO")

    log("Loading TextChunker...", "WAIT")
    chunker = TextChunker()
    log(f"TextChunker ready: method={chunker.method}, size={chunker.chunk_size}", "OK")

    log("Loading TextEmbedding...", "WAIT")
    embedder = TextEmbedding()
    log(f"TextEmbedding ready: dim={embedder.embedding_dim}", "OK")

    log("Connecting to Qdrant...", "WAIT")
    vector_db = VectorDatabase(
        collection_name=os.getenv("COLLECTION_NAME", "knowledge_base"),
        embedding_dimension=embedder.embedding_dim
    )
    log(f"Qdrant connected: collection={vector_db.collection_name}", "OK")

    # ===== STEP 4: Reset Qdrant if requested =====
    # When processing single file, don't reset entire collection
    if single_file and reset_qdrant:
        log("Single file mode: skipping full reset (use --no-reset is implicit)", "INFO")
        reset_qdrant = False

    if reset_qdrant:
        print("\n--- Step 4: Reset Qdrant ---")
        try:
            count = vector_db.client.count(collection_name=vector_db.collection_name).count
            log(f"Deleting {count} existing points...", "WAIT")
            vector_db.delete_collection()
            vector_db.create_collection()
            log("Qdrant reset complete", "OK")
        except Exception as e:
            log(f"Error resetting Qdrant: {e}", "ERROR")
            vector_db.create_collection()

    # ===== STEP 5: Process and index each document =====
    print("\n--- Step 5: Process and Index Documents ---")
    log(f"Method override: {postprocess_method or 'auto (based on extraction type)'}", "INFO")
    if sentence_mode:
        log("Sentence mode: ON (each sentence = 1 chunk)", "INFO")

    total_chunks = 0
    total_docs = len(documents)

    for i, doc in enumerate(documents, 1):
        doc_id = doc['_doc_id']
        file_type = doc.get('file_type', 'unknown')
        content = doc.get('content', '')

        # Get filename from registry
        filename = doc_registry.get(doc_id, {}).get('filename', doc_id)

        if not content:
            log(f"[{i}/{total_docs}] {filename} - SKIPPED (no content)", "WARN")
            continue

        # Use extraction_type from JSON (new format) or fallback to determine_extraction_type (old format)
        extraction_type = doc.get('extraction_type') or determine_extraction_type(file_type)

        print(f"\n{'='*60}")
        log(f"[{i}/{total_docs}] Processing: {filename}", "WAIT")
        log(f"  ID: {doc_id}", "INFO")
        log(f"  Type: {file_type} | Extraction: {extraction_type}", "INFO")
        log(f"  Content: {len(content):,} chars", "INFO")

        # 1. Post-process (using same processor instance)
        log("  Post-processing...", "WAIT")
        pp_start = time.time()

        if postprocess_method:
            processed_content = processor.process(
                content, extraction_type,
                method_override=postprocess_method,
                sentence_mode=sentence_mode
            )
        else:
            processed_content = processor.process(
                content, extraction_type,
                sentence_mode=sentence_mode
            )

        pp_time = time.time() - pp_start
        log(f"  Post-processed: {len(processed_content):,} chars ({pp_time:.1f}s)", "OK")

        # 2. Chunk
        log("  Chunking...", "WAIT")
        chunks = chunker.chunk_text(processed_content)
        chunk_texts = [c["text"] for c in chunks]
        log(f"  Chunks created: {len(chunks)}", "OK")

        if not chunk_texts:
            log("  SKIPPED (no chunks)", "WARN")
            continue

        # 3. Embed
        log("  Embedding...", "WAIT")
        emb_start = time.time()
        embeddings = embedder.encode_text(chunk_texts, show_progress=False)
        emb_time = time.time() - emb_start
        log(f"  Embedded: {len(embeddings)} vectors ({emb_time:.1f}s)", "OK")

        # 4. Prepare for Qdrant
        chunks_with_embeddings = []
        chunk_ids = []

        for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{j:04d}"
            chunk_ids.append(chunk_id)

            chunks_with_embeddings.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                "doc_id": doc_id,
                "chunk_index": j,
                "file_type": file_type,
                "extraction_type": extraction_type,
            })

        # 5. Delete old chunks for this doc (if not full reset)
        if not reset_qdrant:
            try:
                vector_db.delete_by_filter({"doc_id": doc_id})
                log(f"  Deleted old chunks for {doc_id}", "OK")
            except Exception as e:
                log(f"  No old chunks to delete (or error: {e})", "INFO")

        # 6. Upload to Qdrant
        log("  Uploading to Qdrant...", "WAIT")
        vector_db.add_documents(chunks_with_embeddings)
        total_chunks += len(chunks)
        log(f"  Indexed: {len(chunks)} chunks", "OK")

        # Update registry with new chunk info
        if doc_id in doc_registry:
            doc_registry[doc_id]['chunk_ids'] = chunk_ids
            doc_registry[doc_id]['chunk_count'] = len(chunk_ids)
            doc_registry[doc_id]['status'] = 'indexed'
            doc_registry[doc_id]['updated_at'] = datetime.now().isoformat()

    # ===== STEP 6: Save updated index.json =====
    print("\n--- Step 6: Save Index ---")
    index_data = {
        "version": "1.0",
        "updated_at": datetime.now().isoformat(),
        "documents": doc_registry,
        "stats": {
            "total_documents": len(doc_registry),
            "total_chunks": total_chunks,
        }
    }

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    log(f"Index saved: {index_path}", "OK")

    # ===== SUMMARY =====
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  RE-INDEX COMPLETED")
    print("=" * 60)
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", "OK")
    log(f"Documents processed: {len(documents)}", "OK")
    log(f"Total chunks indexed: {total_chunks}", "OK")

    # Verify Qdrant
    try:
        qdrant_count = vector_db.client.count(collection_name=vector_db.collection_name).count
        log(f"Qdrant points verified: {qdrant_count}", "OK")
    except Exception:
        pass

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Re-index processed documents to Qdrant"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes"
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Don't reset Qdrant before indexing"
    )
    parser.add_argument(
        "--method",
        choices=["none", "transformer", "ollama"],
        help="Override post-processing method for all documents"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process only this file (filename in processed/ dir, e.g., 'doc_8786094a.json')"
    )
    parser.add_argument(
        "--sentence-mode",
        action="store_true",
        help="Process each sentence as a separate chunk (more accurate but slower)"
    )

    args = parser.parse_args()

    reindex_documents(
        dry_run=args.dry_run,
        reset_qdrant=not args.no_reset,
        postprocess_method=args.method,
        single_file=args.file,
        sentence_mode=args.sentence_mode
    )


if __name__ == "__main__":
    main()
