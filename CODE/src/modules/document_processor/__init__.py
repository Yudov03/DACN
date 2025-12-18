"""
Document Processor Module
=========================

A comprehensive document processing module that supports 68 file formats:
- PDF (text-based and scanned with OCR)
- Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- Images (.png, .jpg, etc.) with OCR
- Text files (.txt, .md, .csv, etc.)
- Code files (.py, .js, .ts, .java, .cpp, .go, .rs, etc.)
- Audio files (.mp3, .wav, .m4a, etc.) with Whisper ASR
- Video files (.mp4, .avi, .mkv, etc.) with Whisper ASR

Usage:
    from src.modules.document_processor import UnifiedProcessor, process_document

    # Using UnifiedProcessor
    processor = UnifiedProcessor()
    result = processor.process("document.pdf")
    print(result.content)

    # Quick processing
    result = process_document("document.pdf")

    # Batch processing
    results = processor.process_batch(["doc1.pdf", "doc2.docx"])

    # Process audio/video
    result = processor.process("lecture.mp4")
    print(result.content)  # Transcribed text
    for chunk in result.chunks:
        print(f"[{chunk.metadata['start_time']:.1f}s] {chunk.text}")

    # With specific processor
    from src.modules.document_processor import PDFProcessor
    pdf_processor = PDFProcessor({"ocr_enabled": True})
    result = pdf_processor.process("scanned.pdf")
"""

from .base import (
    BaseProcessor,
    ProcessedDocument,
    DocumentMetadata,
    TextChunk,
    Table,
    ImageInfo,
)

from .pdf_processor import PDFProcessor
from .word_processor import WordProcessor
from .excel_processor import ExcelProcessor
from .pptx_processor import PowerPointProcessor
from .image_processor import ImageProcessor
from .text_processor import TextProcessor
from .audio_processor import AudioProcessor, format_transcript_with_timestamps
from .video_processor import VideoProcessor
from .unified_processor import UnifiedProcessor, process_document
from .ocr_engine import OCREngine, ImagePreprocessor, ocr_image


__all__ = [
    # Base classes
    "BaseProcessor",
    "ProcessedDocument",
    "DocumentMetadata",
    "TextChunk",
    "Table",
    "ImageInfo",
    # Processors
    "PDFProcessor",
    "WordProcessor",
    "ExcelProcessor",
    "PowerPointProcessor",
    "ImageProcessor",
    "TextProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "UnifiedProcessor",
    # OCR Engine
    "OCREngine",
    "ImagePreprocessor",
    "ocr_image",
    # Convenience functions
    "process_document",
    "format_transcript_with_timestamps",
]

__version__ = "1.0.0"
