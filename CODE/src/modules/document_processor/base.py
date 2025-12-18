"""
Base classes and data models for document processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time


@dataclass
class DocumentMetadata:
    """Metadata của document"""
    filename: str
    file_size: int  # bytes
    page_count: Optional[int] = None  # for PDF, Word
    sheet_count: Optional[int] = None  # for Excel
    author: Optional[str] = None
    title: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    language: Optional[str] = None  # detected language
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "filename": self.filename,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "sheet_count": self.sheet_count,
            "author": self.author,
            "title": self.title,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
            "language": self.language,
            "extra": self.extra,
        }


@dataclass
class TextChunk:
    """Một chunk text"""
    text: str
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Table:
    """Extracted table"""
    data: List[List[str]]  # 2D array
    headers: Optional[List[str]] = None
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    table_index: Optional[int] = None


@dataclass
class ImageInfo:
    """Image trong document"""
    description: str  # OCR text hoặc caption
    page_number: Optional[int] = None
    image_index: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: Optional[float] = None


@dataclass
class ProcessedDocument:
    """Kết quả sau khi process một file"""

    # Content
    content: str  # Full extracted text
    chunks: List[TextChunk] = field(default_factory=list)  # Pre-chunked text

    # Metadata
    metadata: DocumentMetadata = None
    source_file: str = ""
    file_type: str = ""

    # Additional
    tables: List[Table] = field(default_factory=list)
    images: List[ImageInfo] = field(default_factory=list)

    # Processing info
    processed_at: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0  # seconds
    processor_version: str = "1.0"

    # Status
    success: bool = True
    error_message: Optional[str] = None

    # Extraction type: "direct" (text parsed) or "indirect" (OCR/ASR)
    extraction_type: str = "direct"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "chunks": [{"text": c.text, "page": c.page_number} for c in self.chunks],
            "metadata": {
                "filename": self.metadata.filename if self.metadata else "",
                "file_size": self.metadata.file_size if self.metadata else 0,
                "page_count": self.metadata.page_count if self.metadata else None,
                "sheet_count": self.metadata.sheet_count if self.metadata else None,
            },
            "source_file": self.source_file,
            "file_type": self.file_type,
            "tables_count": len(self.tables),
            "processed_at": self.processed_at.isoformat(),
            "processing_time": self.processing_time,
            "success": self.success,
            "extraction_type": self.extraction_type,
        }


class BaseProcessor(ABC):
    """Abstract base class cho tất cả document processors"""

    VERSION = "1.0"

    def __init__(self, config: Dict = None):
        """
        Initialize processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def process(self, file_path: str) -> ProcessedDocument:
        """
        Process một file và trả về ProcessedDocument.

        Args:
            file_path: Đường dẫn đến file

        Returns:
            ProcessedDocument với content và metadata
        """
        pass

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Trả về list các extension được hỗ trợ (vd: ['.pdf', '.PDF'])"""
        pass

    def validate(self, file_path: str) -> bool:
        """
        Validate file trước khi process.

        Args:
            file_path: Đường dẫn file

        Returns:
            True nếu valid

        Raises:
            FileNotFoundError: File không tồn tại
            ValueError: Extension không hỗ trợ hoặc file quá lớn
        """
        path = Path(file_path)

        # Check exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check extension
        ext = path.suffix.lower()
        supported = [e.lower() for e in self.supported_extensions()]
        if ext not in supported:
            raise ValueError(
                f"Unsupported extension: {ext}. "
                f"Supported: {supported}"
            )

        # Check file size (default max 100MB)
        max_size = self.config.get("max_file_size", 100 * 1024 * 1024)
        if path.stat().st_size > max_size:
            raise ValueError(
                f"File too large: {path.stat().st_size / 1024 / 1024:.1f}MB. "
                f"Max: {max_size / 1024 / 1024:.1f}MB"
            )

        return True

    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract basic metadata từ file.

        Args:
            file_path: Đường dẫn file

        Returns:
            DocumentMetadata with basic info
        """
        path = Path(file_path)
        stat = path.stat()

        return DocumentMetadata(
            filename=path.name,
            file_size=stat.st_size,
            created_date=datetime.fromtimestamp(stat.st_ctime),
            modified_date=datetime.fromtimestamp(stat.st_mtime)
        )

    def _create_error_result(
        self,
        file_path: str,
        error: Exception,
        start_time: float
    ) -> ProcessedDocument:
        """Create error result when processing fails"""
        return ProcessedDocument(
            content="",
            metadata=self.extract_metadata(file_path) if Path(file_path).exists() else None,
            source_file=file_path,
            file_type=Path(file_path).suffix.lower().strip('.'),
            processed_at=datetime.now(),
            processing_time=time.time() - start_time,
            processor_version=self.VERSION,
            success=False,
            error_message=str(error)
        )
