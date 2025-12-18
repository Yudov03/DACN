"""
Image Processor - Extract text from images using OCR.
Supports Vietnamese and English text.

Enhanced OCR using PaddleOCR with advanced preprocessing.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    BaseProcessor,
    ProcessedDocument,
    DocumentMetadata,
    ImageInfo,
)
from .ocr_engine import OCREngine


class ImageProcessor(BaseProcessor):
    """
    Processor cho Image files sử dụng OCR.

    Features:
    - OCR với PaddleOCR (EasyOCR fallback)
    - Hỗ trợ tiếng Việt và tiếng Anh
    - Advanced image preprocessing (deskew, denoise, binarize)
    - Confidence filtering
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Image Processor.

        Args:
            config: Configuration dict with options:
                - language: OCR language (default: 'vi')
                - min_confidence: Min confidence threshold (default: 0.5)
                - use_gpu: Use GPU for OCR (default: True)
                - preprocess: Enable image preprocessing (default: True)
        """
        super().__init__(config)
        self.language = self.config.get("language", os.getenv("OCR_LANGUAGE", "vi"))
        self.min_confidence = self.config.get("min_confidence", float(os.getenv("OCR_MIN_CONFIDENCE", "0.5")))
        self.use_gpu = self.config.get("use_gpu", os.getenv("OCR_USE_GPU", "true").lower() == "true")
        self.preprocess = self.config.get("preprocess", True)
        self._ocr_engine = None

    def supported_extensions(self) -> List[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]

    def _get_ocr_engine(self) -> OCREngine:
        """Lazy load OCR engine (PaddleOCR with EasyOCR fallback)"""
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(
                language=self.language,
                use_gpu=self.use_gpu,
                preprocess=self.preprocess,
                min_confidence=self.min_confidence
            )
        return self._ocr_engine

    def process(self, file_path: str) -> ProcessedDocument:
        """
        Process image file with enhanced OCR.

        Features:
        - PaddleOCR for better Vietnamese accuracy
        - Advanced preprocessing (deskew, denoise, binarize)
        - Confidence filtering
        """
        start_time = time.time()

        try:
            self.validate(file_path)
        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

        try:
            from PIL import Image
        except ImportError:
            return self._create_error_result(
                file_path,
                ImportError("Pillow not installed. Run: pip install Pillow"),
                start_time
            )

        try:
            # Perform OCR using enhanced engine
            ocr = self._get_ocr_engine()
            result = ocr.extract_text(file_path, detailed=True)

            # Extract text and detection info
            text_parts = []
            images_info = []

            for idx, ocr_result in enumerate(result.results):
                text_parts.append(ocr_result.text)
                images_info.append(ImageInfo(
                    description=ocr_result.text,
                    image_index=idx,
                    bbox=ocr_result.bbox,
                    confidence=ocr_result.confidence
                ))

            # Load image for metadata
            img = Image.open(file_path)
            metadata = self._extract_image_metadata(img, file_path)
            metadata.extra["ocr_engine"] = result.engine
            metadata.extra["ocr_confidence"] = result.confidence_avg

            full_content = "\n".join(text_parts)

            return ProcessedDocument(
                content=full_content,
                chunks=[],
                metadata=metadata,
                source_file=file_path,
                file_type="image",
                images=images_info,
                processed_at=datetime.now(),
                processing_time=time.time() - start_time,
                processor_version=self.VERSION,
                success=True,
                extraction_type="indirect"  # OCR always produces indirect extraction
            )

        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

    def _extract_image_metadata(self, img, file_path: str) -> DocumentMetadata:
        """Extract image-specific metadata"""
        base_meta = self.extract_metadata(file_path)

        base_meta.extra = {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
        }

        # Try to get EXIF data
        try:
            exif = img._getexif()
            if exif:
                base_meta.extra["has_exif"] = True
        except Exception:
            pass

        return base_meta
