"""
Unified Processor - Factory class to process any supported file type.
Automatically selects the appropriate processor based on file extension.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseProcessor, ProcessedDocument
from .pdf_processor import PDFProcessor
from .word_processor import WordProcessor
from .excel_processor import ExcelProcessor
from .pptx_processor import PowerPointProcessor
from .image_processor import ImageProcessor
from .text_processor import TextProcessor
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor


class UnifiedProcessor:
    """
    Factory class để process bất kỳ loại file nào.
    Tự động chọn processor phù hợp dựa trên extension.

    Usage:
        processor = UnifiedProcessor()

        # Single file
        result = processor.process("document.pdf")

        # Batch processing
        results = processor.process_batch(["doc1.pdf", "doc2.docx"])

        # With progress callback
        results = processor.process_batch(
            files,
            on_progress=lambda i, total, f: print(f"{i}/{total}: {f}")
        )
    """

    VERSION = "1.0"

    def __init__(self, config: Dict = None):
        """
        Initialize Unified Processor.

        Args:
            config: Configuration dict passed to individual processors
        """
        self.config = config or {}

        # Initialize processors (lazy - only when needed)
        self._processors: Dict[str, BaseProcessor] = {}
        self._processor_classes = {
            # Documents
            ".pdf": PDFProcessor,
            ".docx": WordProcessor,
            ".doc": WordProcessor,
            # Presentations
            ".pptx": PowerPointProcessor,
            ".ppt": PowerPointProcessor,
            # Spreadsheets
            ".xlsx": ExcelProcessor,
            ".xls": ExcelProcessor,
            # Images (OCR)
            ".png": ImageProcessor,
            ".jpg": ImageProcessor,
            ".jpeg": ImageProcessor,
            ".bmp": ImageProcessor,
            ".tiff": ImageProcessor,
            ".tif": ImageProcessor,
            ".webp": ImageProcessor,
            # Text files
            ".txt": TextProcessor,
            ".md": TextProcessor,
            ".csv": TextProcessor,
            ".tsv": TextProcessor,
            ".json": TextProcessor,
            ".xml": TextProcessor,
            ".html": TextProcessor,
            ".log": TextProcessor,
            ".ini": TextProcessor,
            ".cfg": TextProcessor,
            ".rtf": TextProcessor,
            # Code files
            ".py": TextProcessor,
            ".js": TextProcessor,
            ".ts": TextProcessor,
            ".jsx": TextProcessor,
            ".tsx": TextProcessor,
            ".java": TextProcessor,
            ".kt": TextProcessor,
            ".cpp": TextProcessor,
            ".c": TextProcessor,
            ".h": TextProcessor,
            ".hpp": TextProcessor,
            ".go": TextProcessor,
            ".rs": TextProcessor,
            ".rb": TextProcessor,
            ".php": TextProcessor,
            ".swift": TextProcessor,
            ".cs": TextProcessor,
            ".vb": TextProcessor,
            ".sql": TextProcessor,
            ".sh": TextProcessor,
            ".bash": TextProcessor,
            ".ps1": TextProcessor,
            ".yaml": TextProcessor,
            ".yml": TextProcessor,
            ".toml": TextProcessor,
            ".r": TextProcessor,
            ".R": TextProcessor,
            ".scala": TextProcessor,
            # Audio files (Whisper ASR)
            ".mp3": AudioProcessor,
            ".wav": AudioProcessor,
            ".m4a": AudioProcessor,
            ".flac": AudioProcessor,
            ".ogg": AudioProcessor,
            ".wma": AudioProcessor,
            ".aac": AudioProcessor,
            # Video files (Extract audio + Whisper ASR)
            ".mp4": VideoProcessor,
            ".avi": VideoProcessor,
            ".mkv": VideoProcessor,
            ".mov": VideoProcessor,
            ".wmv": VideoProcessor,
            ".flv": VideoProcessor,
            ".webm": VideoProcessor,
            ".m4v": VideoProcessor,
        }

    def _get_processor(self, extension: str) -> BaseProcessor:
        """Get or create processor for extension"""
        ext = extension.lower()
        if ext not in self._processors:
            if ext not in self._processor_classes:
                raise ValueError(f"Unsupported file type: {ext}")
            self._processors[ext] = self._processor_classes[ext](self.config)
        return self._processors[ext]

    def process(self, file_path: str) -> ProcessedDocument:
        """
        Process một file bất kỳ.

        Args:
            file_path: Đường dẫn đến file

        Returns:
            ProcessedDocument

        Raises:
            ValueError: Nếu file type không được hỗ trợ
            FileNotFoundError: Nếu file không tồn tại
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext not in self._processor_classes:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(self._processor_classes.keys())}"
            )

        processor = self._get_processor(ext)
        return processor.process(file_path)

    def process_batch(
        self,
        file_paths: List[str],
        on_progress: Callable[[int, int, str], None] = None,
        on_error: Callable[[str, Exception], None] = None,
        parallel: bool = False,
        max_workers: int = 4
    ) -> List[ProcessedDocument]:
        """
        Process nhiều files.

        Args:
            file_paths: List đường dẫn files
            on_progress: Callback(current, total, file_path) for progress updates
            on_error: Callback(file_path, error) when processing fails
            parallel: Use parallel processing (default: False)
            max_workers: Max parallel workers (default: 4)

        Returns:
            List of ProcessedDocument (None for failed files)
        """
        total = len(file_paths)
        results = [None] * total

        if parallel:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.process, fp): idx
                    for idx, fp in enumerate(file_paths)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    file_path = file_paths[idx]
                    completed += 1

                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        if on_error:
                            on_error(file_path, e)
                        results[idx] = None

                    if on_progress:
                        on_progress(completed, total, file_path)
        else:
            # Sequential processing
            for idx, file_path in enumerate(file_paths):
                try:
                    results[idx] = self.process(file_path)
                except Exception as e:
                    if on_error:
                        on_error(file_path, e)
                    results[idx] = None

                if on_progress:
                    on_progress(idx + 1, total, file_path)

        return results

    def supported_extensions(self) -> List[str]:
        """Trả về tất cả extensions được hỗ trợ"""
        return list(self._processor_classes.keys())

    def get_processor(self, extension: str) -> Optional[BaseProcessor]:
        """
        Lấy processor cho một extension cụ thể.

        Args:
            extension: File extension (e.g., '.pdf')

        Returns:
            BaseProcessor or None if not supported
        """
        ext = extension.lower()
        if ext not in self._processor_classes:
            return None
        return self._get_processor(ext)

    def is_supported(self, file_path: str) -> bool:
        """
        Check if file type is supported.

        Args:
            file_path: Path to file

        Returns:
            True if supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in self._processor_classes

    @classmethod
    def get_supported_types(cls) -> Dict[str, List[str]]:
        """
        Get supported file types grouped by category.

        Returns:
            Dict with categories and their extensions
        """
        return {
            "documents": [".pdf", ".docx", ".doc"],
            "presentations": [".pptx", ".ppt"],
            "spreadsheets": [".xlsx", ".xls"],
            "images": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"],
            "text": [".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".log", ".ini", ".cfg", ".rtf"],
            "code": [
                ".py", ".js", ".ts", ".jsx", ".tsx",
                ".java", ".kt", ".cpp", ".c", ".h", ".hpp",
                ".go", ".rs", ".rb", ".php", ".swift",
                ".cs", ".vb", ".sql", ".sh", ".bash", ".ps1",
                ".yaml", ".yml", ".toml", ".r", ".R", ".scala"
            ],
            "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"],
            "video": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"],
        }


# Convenience function
def process_document(file_path: str, config: Dict = None) -> ProcessedDocument:
    """
    Quick function to process a single document.

    Args:
        file_path: Path to file
        config: Optional configuration

    Returns:
        ProcessedDocument
    """
    processor = UnifiedProcessor(config)
    return processor.process(file_path)
