"""
Knowledge Base Manager
======================

Quản lý tài liệu cho RAG system: upload, index, delete, search, export/import.

Usage:
    from src.modules.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()

    # Add document
    doc_id = kb.add_document("path/to/file.pdf", tags=["important"])

    # List documents
    docs = kb.list_documents(filter={"file_type": "pdf"})

    # Remove document
    kb.remove_document(doc_id)

    # Export/Import
    kb.export_kb("backup.zip")
    kb.import_kb("backup.zip")
"""

import json
import uuid
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
import logging

from .post_processing import post_process_text
from .file_types import (
    TYPE_TO_FOLDER,
    MEDIA_TYPES,
    get_folder_for_type,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class DocumentInfo:
    """Thông tin một document trong Knowledge Base"""

    # Identifiers
    id: str
    filename: str

    # Paths (relative to base_dir)
    original_path: str
    processed_path: str

    # File info
    file_type: str
    file_size: int
    file_hash: str

    # Content info
    page_count: Optional[int] = None
    chunk_count: int = 0
    chunk_ids: List[str] = field(default_factory=list)

    # User metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None

    # Timestamps
    uploaded_at: str = ""
    processed_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Status: pending, processing, indexed, error
    status: str = "pending"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentInfo':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class KBStats:
    """Thống kê Knowledge Base"""

    total_documents: int = 0
    total_chunks: int = 0
    total_size_bytes: int = 0

    documents_by_type: Dict[str, int] = field(default_factory=dict)
    documents_by_status: Dict[str, int] = field(default_factory=dict)

    last_upload: Optional[str] = None
    last_modified: Optional[str] = None

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)


# ==============================================================================
# Exceptions
# ==============================================================================

class KnowledgeBaseError(Exception):
    """Base exception for Knowledge Base errors"""
    pass


class DocumentNotFoundError(KnowledgeBaseError):
    """Document not found in Knowledge Base"""
    pass


class DuplicateDocumentError(KnowledgeBaseError):
    """Document already exists in Knowledge Base"""
    pass


class ProcessingError(KnowledgeBaseError):
    """Error during document processing"""
    pass


# ==============================================================================
# Document Registry
# ==============================================================================

class DocumentRegistry:
    """
    Manages the index.json file that tracks all documents in the Knowledge Base.
    """

    VERSION = "1.0"

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.documents: Dict[str, DocumentInfo] = {}
        self._load()

    def _load(self):
        """Load registry from index.json"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.documents = {
                    doc_id: DocumentInfo.from_dict(doc_data)
                    for doc_id, doc_data in data.get("documents", {}).items()
                }
                logger.info(f"Loaded {len(self.documents)} documents from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.documents = {}
        else:
            self.documents = {}

    def save(self):
        """Save registry to index.json"""
        data = {
            "version": self.VERSION,
            "updated_at": datetime.now().isoformat(),
            "documents": {
                doc_id: doc.to_dict()
                for doc_id, doc in self.documents.items()
            },
            "stats": {
                "total_documents": len(self.documents),
                "total_chunks": sum(d.chunk_count for d in self.documents.values()),
            }
        }

        # Ensure parent directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, doc_info: DocumentInfo) -> str:
        """Add document to registry"""
        self.documents[doc_info.id] = doc_info
        self.save()
        return doc_info.id

    def remove(self, doc_id: str) -> bool:
        """Remove document from registry"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.save()
            return True
        return False

    def get(self, doc_id: str) -> Optional[DocumentInfo]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def update(self, doc_id: str, updates: Dict) -> bool:
        """Update document info"""
        if doc_id not in self.documents:
            return False

        doc = self.documents[doc_id]
        for key, value in updates.items():
            if hasattr(doc, key):
                setattr(doc, key, value)

        doc.updated_at = datetime.now().isoformat()
        self.save()
        return True

    def find_by_hash(self, file_hash: str) -> Optional[DocumentInfo]:
        """Find document by file hash"""
        for doc in self.documents.values():
            if doc.file_hash == file_hash:
                return doc
        return None

    def list_all(self) -> List[DocumentInfo]:
        """List all documents"""
        return list(self.documents.values())

    def filter(
        self,
        file_type: str = None,
        status: str = None,
        category: str = None,
        tags: List[str] = None
    ) -> List[DocumentInfo]:
        """Filter documents"""
        results = list(self.documents.values())

        if file_type:
            results = [d for d in results if d.file_type == file_type]

        if status:
            results = [d for d in results if d.status == status]

        if category:
            results = [d for d in results if d.category == category]

        if tags:
            results = [d for d in results if all(t in d.tags for t in tags)]

        return results

    def search(self, query: str) -> List[DocumentInfo]:
        """Search documents by filename or metadata"""
        query_lower = query.lower()
        results = []

        for doc in self.documents.values():
            # Search in filename
            if query_lower in doc.filename.lower():
                results.append(doc)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in doc.tags):
                results.append(doc)
                continue

            # Search in category
            if doc.category and query_lower in doc.category.lower():
                results.append(doc)
                continue

            # Search in metadata
            for value in doc.metadata.values():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(doc)
                    break

        return results


# ==============================================================================
# Knowledge Base
# ==============================================================================

class KnowledgeBase:
    """
    Knowledge Base Manager for RAG system.

    Supports:
    - Upload documents (PDF, Word, Excel, Image, Text)
    - Index documents for semantic search
    - Delete documents
    - Export/Import Knowledge Base
    """

    # File type to folder mapping - imported from file_types module
    # Use TYPE_TO_FOLDER from src.modules.file_types for the mapping

    def __init__(
        self,
        base_dir: str = "./data/knowledge_base",
        vector_db = None,
        embedder = None,
        chunker = None,
        processor = None,
        config: Dict = None
    ):
        """
        Initialize Knowledge Base.

        Args:
            base_dir: Base directory for Knowledge Base
            vector_db: VectorDatabase instance (optional, will create if None)
            embedder: TextEmbedding instance (optional, will create if None)
            chunker: TextChunker instance (optional, will create if None)
            processor: UnifiedProcessor instance (optional, will create if None)
            config: Additional configuration
        """
        self.base_dir = Path(base_dir)
        self.config = config or {}

        # Create directories
        self._init_directories()

        # Initialize registry
        self.registry = DocumentRegistry(self.base_dir / "index.json")

        # Initialize components (lazy loading)
        self._vector_db = vector_db
        self._embedder = embedder
        self._chunker = chunker
        self._processor = processor

    def _init_directories(self):
        """Create required directories"""
        dirs = [
            self.base_dir / "documents" / "pdf",
            self.base_dir / "documents" / "word",
            self.base_dir / "documents" / "excel",
            self.base_dir / "documents" / "image",
            self.base_dir / "documents" / "text",
            self.base_dir / "processed",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def vector_db(self):
        """Lazy load VectorDatabase"""
        if self._vector_db is None:
            from src.modules import VectorDatabase
            collection_name = self.config.get("collection_name", "knowledge_base")
            # Get embedding dimension from embedder
            embedding_dim = self.embedder.embedding_dim
            self._vector_db = VectorDatabase(
                collection_name=collection_name,
                embedding_dimension=embedding_dim
            )
        return self._vector_db

    @property
    def embedder(self):
        """Lazy load TextEmbedding"""
        if self._embedder is None:
            from src.modules import TextEmbedding
            self._embedder = TextEmbedding()
        return self._embedder

    @property
    def chunker(self):
        """Lazy load TextChunker"""
        if self._chunker is None:
            import os
            from src.modules import TextChunker
            self._chunker = TextChunker(
                # Read from env for consistency with chunking_module.py
                chunk_size=self.config.get("chunk_size", int(os.getenv("CHUNK_SIZE", "500"))),
                chunk_overlap=self.config.get("chunk_overlap", int(os.getenv("CHUNK_OVERLAP", "50"))),
                method=self.config.get("chunking_method", os.getenv("CHUNKING_METHOD", "semantic")),
                embedding_provider="local"  # Use local embeddings (SBERT/E5)
            )
        return self._chunker

    @property
    def processor(self):
        """Lazy load UnifiedProcessor"""
        if self._processor is None:
            from src.modules.document_processor import UnifiedProcessor
            self._processor = UnifiedProcessor()
        return self._processor

    # ==========================================================================
    # Document Management
    # ==========================================================================

    def add_document(
        self,
        file_path: str,
        metadata: Dict = None,
        tags: List[str] = None,
        category: str = None,
        skip_duplicate: bool = True
    ) -> str:
        """
        Add a document to the Knowledge Base.

        Args:
            file_path: Path to the document file
            metadata: Additional metadata
            tags: Tags for the document
            category: Category/folder name
            skip_duplicate: If True, skip if duplicate exists; if False, raise error

        Returns:
            doc_id: ID of the added document

        Raises:
            FileNotFoundError: If file doesn't exist
            DuplicateDocumentError: If duplicate exists and skip_duplicate=False
            ProcessingError: If processing fails
        """
        file_path = Path(file_path)

        # 1. Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 2. Check duplicate by hash
        file_hash = self._compute_hash(file_path)
        existing = self.registry.find_by_hash(file_hash)
        if existing:
            if skip_duplicate:
                logger.info(f"Document already exists: {existing.id}")
                return existing.id
            else:
                raise DuplicateDocumentError(
                    f"Document already exists with ID: {existing.id}"
                )

        # 3. Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # 4. Determine file type and destination
        file_type = file_path.suffix.lower().strip('.')
        folder = get_folder_for_type(file_type)
        dest_dir = self.base_dir / "documents" / folder
        dest_path = dest_dir / f"{doc_id}{file_path.suffix}"

        # 5. Copy file to Knowledge Base (create folder if not exists)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)
        logger.info(f"Copied file to: {dest_path}")

        try:
            # 6. Process document
            processed = self.processor.process(str(dest_path))

            if not processed.success:
                raise ProcessingError(
                    f"Failed to process document: {processed.error_message}"
                )

            # 7. Save processed result
            processed_path = self.base_dir / "processed" / f"{doc_id}.json"
            self._save_processed(processed, processed_path)

            # 7b. Save transcript for audio/video files
            if file_type in MEDIA_TYPES and processed.content:
                transcript_dir = self.base_dir / "transcripts"
                transcript_dir.mkdir(parents=True, exist_ok=True)
                transcript_path = transcript_dir / f"{doc_id}.txt"
                # Handle encoding issues with filename
                safe_filename = file_path.name.encode('utf-8', errors='replace').decode('utf-8')
                with open(transcript_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"# Transcript: {safe_filename}\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                    f.write(processed.content)
                logger.info(f"Saved transcript to: {transcript_path}")

            # 8. Post-process text (fix OCR/ASR errors) BEFORE chunking
            # Use extraction_type from processor (determined during processing)
            # - "direct": text parsed directly (Word, txt, PDF with text)
            # - "indirect": OCR/ASR was used (scanned PDF, images, audio, video)
            extraction_type = processed.extraction_type

            processed_content = post_process_text(processed.content, extraction_type=extraction_type)
            logger.info(f"Post-processed content with extraction_type={extraction_type}")
            logger.debug(f"Post-processed content length: {len(processed_content)}")

            # 9. Chunk text
            chunks = self.chunker.chunk_text(processed_content)
            chunk_texts = [c["text"] for c in chunks]

            # 10. Generate embeddings and index
            chunk_ids = []
            if chunk_texts:
                embeddings = self.embedder.encode_text(chunk_texts, show_progress=False)

                # Prepare chunks with embeddings for VectorDatabase
                chunks_with_embeddings = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = f"{doc_id}_chunk_{i:04d}"
                    chunk_ids.append(chunk_id)

                    chunks_with_embeddings.append({
                        "chunk_id": chunk_id,  # Store as metadata, not as point ID
                        "text": chunk["text"],
                        "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "start": chunk.get("start", 0),
                        "end": chunk.get("end", len(chunk["text"])),
                    })

                # Add all chunks at once (let Qdrant generate UUIDs)
                self.vector_db.add_documents(chunks_with_embeddings)

            # 11. Create DocumentInfo
            doc_info = DocumentInfo(
                id=doc_id,
                filename=file_path.name,
                original_path=str(dest_path.relative_to(self.base_dir)),
                processed_path=str(processed_path.relative_to(self.base_dir)),
                file_type=file_type,
                file_size=file_path.stat().st_size,
                file_hash=file_hash,
                page_count=processed.metadata.page_count if processed.metadata else None,
                chunk_count=len(chunk_ids),
                chunk_ids=chunk_ids,
                metadata=metadata or {},
                tags=tags or [],
                category=category,
                uploaded_at=datetime.now().isoformat(),
                processed_at=datetime.now().isoformat(),
                status="indexed"
            )

            # 12. Add to registry
            self.registry.add(doc_info)

            logger.info(f"Added document: {doc_id} ({len(chunk_ids)} chunks)")
            return doc_id

        except Exception as e:
            # Cleanup on error
            if dest_path.exists():
                dest_path.unlink()
            raise ProcessingError(f"Failed to add document: {e}")

    def add_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        on_progress: Callable[[int, int, str], None] = None,
        skip_errors: bool = True
    ) -> List[str]:
        """
        Add all supported documents from a folder.

        Args:
            folder_path: Path to folder
            recursive: Scan subfolders
            on_progress: Callback(current, total, filename)
            skip_errors: Continue on errors

        Returns:
            List of added document IDs
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find all supported files
        supported_extensions = list(TYPE_TO_FOLDER.keys())
        files = []

        pattern = "**/*" if recursive else "*"
        for ext in supported_extensions:
            files.extend(folder_path.glob(f"{pattern}.{ext}"))

        # Add each file
        doc_ids = []
        total = len(files)

        for i, file_path in enumerate(files):
            if on_progress:
                on_progress(i + 1, total, file_path.name)

            try:
                doc_id = self.add_document(str(file_path))
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error adding {file_path}: {e}")
                if not skip_errors:
                    raise

        return doc_ids

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the Knowledge Base.

        Args:
            doc_id: Document ID

        Returns:
            True if removed successfully

        Raises:
            DocumentNotFoundError: If document doesn't exist
        """
        doc_info = self.registry.get(doc_id)
        if not doc_info:
            raise DocumentNotFoundError(f"Document not found: {doc_id}")

        # 1. Remove chunks from vector DB by doc_id filter
        try:
            self.vector_db.delete_by_filter({"doc_id": doc_id})
        except Exception as e:
            logger.warning(f"Error deleting chunks for {doc_id}: {e}")

        # 2. Delete original file
        original_path = self.base_dir / doc_info.original_path
        if original_path.exists():
            original_path.unlink()

        # 3. Delete processed file
        processed_path = self.base_dir / doc_info.processed_path
        if processed_path.exists():
            processed_path.unlink()

        # 4. Delete transcript file (for audio/video)
        transcript_path = self.base_dir / "transcripts" / f"{doc_id}.txt"
        if transcript_path.exists():
            transcript_path.unlink()

        # 5. Remove from registry
        self.registry.remove(doc_id)

        logger.info(f"Removed document: {doc_id}")
        return True

    def clear_all(self) -> dict:
        """
        Clear all documents from Knowledge Base and Qdrant.

        Returns:
            dict with counts of deleted items
        """
        result = {
            "documents": 0,
            "chunks": 0,
            "files": 0
        }

        # 1. Get all document IDs
        doc_ids = list(self.registry.documents.keys())
        result["documents"] = len(doc_ids)

        # 2. Clear Qdrant collection (delete and recreate)
        try:
            count_before = self.vector_db.client.count(
                collection_name=self.vector_db.collection_name
            ).count
            self.vector_db.delete_collection()
            result["chunks"] = count_before

            # Recreate empty collection
            self.vector_db.create_collection()
            logger.info(f"Cleared Qdrant: {count_before} points, recreated collection")
        except Exception as e:
            logger.warning(f"Error clearing Qdrant: {e}")

        # 3. Delete all files in documents/, processed/, transcripts/
        for folder in ["documents", "processed", "transcripts"]:
            folder_path = self.base_dir / folder
            if folder_path.exists():
                for f in folder_path.rglob("*"):
                    if f.is_file():
                        try:
                            f.unlink()
                            result["files"] += 1
                        except Exception as e:
                            logger.warning(f"Error deleting {f}: {e}")

        # 4. Reset registry
        self.registry.documents = {}
        self.registry.save()

        logger.info(f"Cleared all: {result}")
        return result

    def update_document(
        self,
        doc_id: str,
        file_path: str = None,
        metadata: Dict = None,
        tags: List[str] = None,
        category: str = None
    ) -> bool:
        """
        Update a document (re-process if new file provided).

        Args:
            doc_id: Document ID
            file_path: New file path (None = just update metadata)
            metadata: New metadata (merged with existing)
            tags: New tags (replaces existing)
            category: New category

        Returns:
            True if updated successfully
        """
        doc_info = self.registry.get(doc_id)
        if not doc_info:
            raise DocumentNotFoundError(f"Document not found: {doc_id}")

        # If new file provided, re-process
        if file_path:
            # Store old info
            old_tags = doc_info.tags
            old_category = doc_info.category
            old_metadata = doc_info.metadata

            # Remove old document
            self.remove_document(doc_id)

            # Add new document with same ID... actually, generate new ID
            new_doc_id = self.add_document(
                file_path,
                metadata=metadata or old_metadata,
                tags=tags or old_tags,
                category=category or old_category
            )
            return True

        # Just update metadata
        updates = {"updated_at": datetime.now().isoformat()}

        if metadata:
            doc_info.metadata.update(metadata)
            updates["metadata"] = doc_info.metadata

        if tags is not None:
            updates["tags"] = tags

        if category is not None:
            updates["category"] = category

        return self.registry.update(doc_id, updates)

    # ==========================================================================
    # Query Methods
    # ==========================================================================

    def get_document(self, doc_id: str) -> DocumentInfo:
        """Get document by ID"""
        doc_info = self.registry.get(doc_id)
        if not doc_info:
            raise DocumentNotFoundError(f"Document not found: {doc_id}")
        return doc_info

    def list_documents(
        self,
        filter: Dict = None,
        sort_by: str = "uploaded_at",
        descending: bool = True,
        limit: int = None
    ) -> List[DocumentInfo]:
        """
        List documents with optional filtering.

        Args:
            filter: Filter criteria
                - file_type: "pdf", "docx", etc.
                - status: "indexed", "pending", "error"
                - category: category name
                - tags: list of tags (AND)
            sort_by: Field to sort by
            descending: Sort order
            limit: Max results

        Returns:
            List of DocumentInfo
        """
        if filter:
            docs = self.registry.filter(**filter)
        else:
            docs = self.registry.list_all()

        # Sort
        if sort_by and hasattr(docs[0] if docs else DocumentInfo, sort_by):
            docs.sort(
                key=lambda d: getattr(d, sort_by) or "",
                reverse=descending
            )

        # Limit
        if limit:
            docs = docs[:limit]

        return docs

    def search_documents(self, query: str, limit: int = 10) -> List[DocumentInfo]:
        """
        Search documents by filename or metadata (text search, not semantic).

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching DocumentInfo
        """
        results = self.registry.search(query)
        return results[:limit] if limit else results

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_stats(self) -> KBStats:
        """Get Knowledge Base statistics"""
        docs = self.registry.list_all()

        # Count by type
        by_type = {}
        for doc in docs:
            by_type[doc.file_type] = by_type.get(doc.file_type, 0) + 1

        # Count by status
        by_status = {}
        for doc in docs:
            by_status[doc.status] = by_status.get(doc.status, 0) + 1

        # Total size
        total_size = sum(doc.file_size for doc in docs)

        # Total chunks
        total_chunks = sum(doc.chunk_count for doc in docs)

        # Last upload
        last_upload = None
        if docs:
            sorted_docs = sorted(docs, key=lambda d: d.uploaded_at or "", reverse=True)
            last_upload = sorted_docs[0].uploaded_at

        return KBStats(
            total_documents=len(docs),
            total_chunks=total_chunks,
            total_size_bytes=total_size,
            documents_by_type=by_type,
            documents_by_status=by_status,
            last_upload=last_upload,
            last_modified=datetime.now().isoformat()
        )

    # ==========================================================================
    # Export / Import
    # ==========================================================================

    def export_kb(self, output_path: str) -> str:
        """
        Export entire Knowledge Base to a zip file.

        Args:
            output_path: Path for output zip file

        Returns:
            Path to created zip file
        """
        output_path = Path(output_path)

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add index.json
            index_path = self.base_dir / "index.json"
            if index_path.exists():
                zf.write(index_path, "index.json")

            # Add documents folder
            docs_dir = self.base_dir / "documents"
            for file_path in docs_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.base_dir)
                    zf.write(file_path, arcname)

            # Add processed folder
            proc_dir = self.base_dir / "processed"
            for file_path in proc_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.base_dir)
                    zf.write(file_path, arcname)

        logger.info(f"Exported Knowledge Base to: {output_path}")
        return str(output_path)

    def import_kb(self, zip_path: str, merge: bool = False) -> bool:
        """
        Import Knowledge Base from a zip file.

        Args:
            zip_path: Path to zip file
            merge: If True, merge with existing; if False, replace

        Returns:
            True if successful
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        # If not merging, clear existing
        if not merge:
            # Remove all documents
            for doc in self.registry.list_all():
                try:
                    self.remove_document(doc.id)
                except Exception as e:
                    logger.warning(f"Error removing {doc.id}: {e}")

        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.base_dir)

        # Reload registry
        self.registry = DocumentRegistry(self.base_dir / "index.json")

        # Re-index if needed
        # TODO: Rebuild vector index for imported documents

        logger.info(f"Imported Knowledge Base from: {zip_path}")
        return True

    def rebuild_index(
        self,
        on_progress: Callable[[int, int, str], None] = None
    ) -> bool:
        """
        Rebuild vector index from processed files.
        Use when index is corrupted or embedding model changed.

        Args:
            on_progress: Callback(current, total, doc_id)

        Returns:
            True if successful
        """
        docs = self.registry.list_all()
        total = len(docs)

        for i, doc in enumerate(docs):
            if on_progress:
                on_progress(i + 1, total, doc.id)

            try:
                # Load processed content
                processed_path = self.base_dir / doc.processed_path
                if not processed_path.exists():
                    logger.warning(f"Processed file not found: {processed_path}")
                    continue

                with open(processed_path, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)

                content = processed_data.get("content", "")
                if not content:
                    continue

                # Delete old chunks by doc_id
                try:
                    self.vector_db.delete_by_filter({"doc_id": doc.id})
                except Exception:
                    pass

                # Re-chunk and re-index
                chunks = self.chunker.chunk_text(content)
                chunk_texts = [c["text"] for c in chunks]

                if chunk_texts:
                    embeddings = self.embedder.encode_text(chunk_texts, show_progress=False)

                    new_chunk_ids = []
                    chunks_with_embeddings = []
                    for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        chunk_id = f"{doc.id}_chunk_{j:04d}"
                        new_chunk_ids.append(chunk_id)

                        chunks_with_embeddings.append({
                            "chunk_id": chunk_id,
                            "text": chunk["text"],
                            "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                            "doc_id": doc.id,
                            "chunk_index": j,
                        })

                    self.vector_db.add_documents(chunks_with_embeddings)

                    # Update registry
                    self.registry.update(doc.id, {
                        "chunk_ids": new_chunk_ids,
                        "chunk_count": len(new_chunk_ids),
                        "status": "indexed"
                    })

            except Exception as e:
                logger.error(f"Error rebuilding index for {doc.id}: {e}")
                self.registry.update(doc.id, {
                    "status": "error",
                    "error_message": str(e)
                })

        logger.info("Index rebuild completed")
        return True

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _save_processed(self, processed, output_path: Path):
        """Save processed document to JSON"""
        data = {
            "content": processed.content,
            "file_type": processed.file_type,
            "extraction_type": processed.extraction_type,  # direct or indirect
            "metadata": processed.metadata.to_dict() if hasattr(processed.metadata, 'to_dict') else {},
            "tables_count": len(processed.tables) if processed.tables else 0,
            "processed_at": datetime.now().isoformat(),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
