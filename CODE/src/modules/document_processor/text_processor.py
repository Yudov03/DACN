"""
Text Processor - Process plain text and markdown files.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .base import (
    BaseProcessor,
    ProcessedDocument,
    DocumentMetadata,
    TextChunk,
)


class TextProcessor(BaseProcessor):
    """
    Processor cho plain text files (.txt, .md, .csv, .json, etc.).

    Features:
    - Read text content
    - Detect encoding
    - Split into chunks by paragraph or lines
    """

    # Common text encodings to try
    ENCODINGS = ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1', 'ascii']

    def __init__(self, config: Dict = None):
        """
        Initialize Text Processor.

        Args:
            config: Configuration dict with options:
                - encoding: Force specific encoding (default: auto-detect)
                - chunk_by: How to chunk - 'paragraph' or 'lines' (default: 'paragraph')
                - min_chunk_size: Min chars per chunk (default: 100)
        """
        super().__init__(config)
        self.encoding = self.config.get("encoding", None)
        self.chunk_by = self.config.get("chunk_by", "paragraph")
        self.min_chunk_size = self.config.get("min_chunk_size", 100)

    def supported_extensions(self) -> List[str]:
        return [
            # Text files
            ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".log", ".ini", ".cfg", ".rtf",
            # Code files
            ".py", ".js", ".ts", ".jsx", ".tsx",  # Python, JavaScript, TypeScript
            ".java", ".kt",                        # Java, Kotlin
            ".cpp", ".c", ".h", ".hpp",            # C/C++
            ".go", ".rs",                          # Go, Rust
            ".rb", ".php", ".swift",               # Ruby, PHP, Swift
            ".cs", ".vb",                          # C#, Visual Basic
            ".sql", ".sh", ".bash", ".ps1",        # SQL, Shell, PowerShell
            ".yaml", ".yml", ".toml",              # Config files
            ".r", ".R", ".scala",                  # R, Scala
        ]

    def process(self, file_path: str) -> ProcessedDocument:
        """Process text file"""
        start_time = time.time()

        try:
            self.validate(file_path)
        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

        try:
            # Read file with encoding detection
            content = self._read_with_encoding(file_path)

            # Create chunks
            chunks = self._create_chunks(content)

            # Metadata
            metadata = self.extract_metadata(file_path)
            metadata.extra = {
                "line_count": content.count('\n') + 1,
                "word_count": len(content.split()),
                "char_count": len(content),
            }

            return ProcessedDocument(
                content=content,
                chunks=chunks,
                metadata=metadata,
                source_file=file_path,
                file_type=Path(file_path).suffix.lower().strip('.'),
                processed_at=datetime.now(),
                processing_time=time.time() - start_time,
                processor_version=self.VERSION,
                success=True
            )

        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

    def _read_with_encoding(self, file_path: str) -> str:
        """Read file with automatic encoding detection"""
        if self.encoding:
            # Use specified encoding
            with open(file_path, 'r', encoding=self.encoding) as f:
                return f.read()

        # Try different encodings
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Last resort: read as binary and decode with errors='replace'
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')

    def _create_chunks(self, content: str) -> List[TextChunk]:
        """Create chunks from content"""
        chunks = []
        char_offset = 0

        if self.chunk_by == "paragraph":
            # Split by double newline (paragraphs)
            parts = content.split('\n\n')
        else:
            # Split by single newline (lines)
            parts = content.split('\n')

        chunk_index = 0
        for part in parts:
            part = part.strip()
            if len(part) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=part,
                    start_char=char_offset,
                    end_char=char_offset + len(part),
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            char_offset += len(part) + 2  # +2 for newlines

        return chunks
