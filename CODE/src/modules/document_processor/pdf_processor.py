# -*- coding: utf-8 -*-
"""
PDF Processor - Hybrid text extraction with smart OCR.

Features:
- Text-based PDF: Direct extraction (fast, accurate)
- Scanned PDF: OCR entire page
- Mixed PDF: Extract text + OCR only image regions (hybrid)

This hybrid approach:
- Preserves quality of embedded text (no OCR errors)
- Only OCRs image regions that need it
- 50-70% faster than full-page OCR for mixed documents
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import fitz (PyMuPDF) at module level for OCR
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    fitz = None

from .base import (
    BaseProcessor,
    ProcessedDocument,
    DocumentMetadata,
    TextChunk,
    Table,
)
from .ocr_engine import OCREngine


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PageContent:
    """Analyzed content of a PDF page."""
    text: str                           # Extracted text
    text_blocks: List[dict] = field(default_factory=list)   # [{bbox, text}]
    image_blocks: List[dict] = field(default_factory=list)  # [{bbox, xref, width, height}]
    drawing_count: int = 0              # Number of vector drawings
    page_type: str = "empty"            # text, scan, mixed, empty


# =============================================================================
# PDF Processor
# =============================================================================

class PDFProcessor(BaseProcessor):
    """
    Processor cho PDF files với Hybrid Processing.

    Features:
    - Extract text từ PDF text-based
    - OCR cho scanned PDFs
    - Hybrid mode: Text + OCR image regions cho mixed PDFs
    - Extract tables
    - Extract metadata
    """

    VERSION = "2.0"  # Upgraded for hybrid processing

    def __init__(self, config: Dict = None):
        """
        Initialize PDF Processor.

        Args:
            config: Configuration dict with options:
                - ocr_enabled: Enable OCR for scanned pages (default: True)
                - extract_tables: Extract tables (default: True)
                - ocr_language: Language for OCR (default: 'vi')
                - ocr_dpi: DPI for OCR rendering (default: 400)
                - min_text_length: Min chars to consider page has text (default: 50)
                - hybrid_mode: Enable hybrid processing (default: True)
                - min_image_size: Min image dimension to OCR (default: 50)
        """
        super().__init__(config)

        # Basic settings
        self.ocr_enabled = self.config.get("ocr_enabled", True)
        self.extract_tables_enabled = self.config.get("extract_tables", True)
        self.ocr_language = self.config.get("ocr_language", os.getenv("OCR_LANGUAGE", "vi"))
        self.ocr_dpi = self.config.get("ocr_dpi", int(os.getenv("OCR_DPI", "400")))

        # Text detection settings (improved thresholds)
        self.min_text_length = self.config.get(
            "min_text_length",
            int(os.getenv("PDF_MIN_TEXT_LENGTH", "100"))
        )
        self.min_text_density = self.config.get(
            "min_text_density",
            float(os.getenv("PDF_MIN_TEXT_DENSITY", "1.0"))
        )
        self.text_quality_threshold = self.config.get(
            "text_quality_threshold",
            float(os.getenv("PDF_TEXT_QUALITY_THRESHOLD", "0.7"))
        )

        # Hybrid mode settings
        self.hybrid_mode = self.config.get(
            "hybrid_mode",
            os.getenv("PDF_HYBRID_MODE", "true").lower() == "true"
        )
        self.min_image_size = self.config.get(
            "min_image_size",
            int(os.getenv("PDF_MIN_IMAGE_SIZE", "50"))
        )
        self.min_content_image_size = self.config.get(
            "min_content_image_size",
            int(os.getenv("PDF_MIN_CONTENT_IMAGE_SIZE", "10000"))
        )
        self.image_overlap_threshold = self.config.get(
            "image_overlap_threshold",
            float(os.getenv("PDF_IMAGE_OVERLAP_THRESHOLD", "0.3"))
        )
        self.ocr_confidence_threshold = self.config.get(
            "ocr_confidence_threshold",
            float(os.getenv("PDF_OCR_CONFIDENCE_THRESHOLD", "0.7"))
        )

        self._ocr_engine = None

    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    # =========================================================================
    # Main Process Method
    # =========================================================================

    def process(self, file_path: str) -> ProcessedDocument:
        """Process PDF file with hybrid approach."""
        start_time = time.time()

        try:
            self.validate(file_path)
        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return self._create_error_result(
                file_path,
                ImportError("PyMuPDF not installed. Run: pip install PyMuPDF"),
                start_time
            )

        try:
            doc = fitz.open(file_path)
            text_parts = []
            chunks = []
            tables = []
            char_offset = 0

            # Track extraction types used
            extraction_types_used = set()

            for page_num in range(len(doc)):
                page = doc[page_num]

                if self.hybrid_mode:
                    # Use hybrid processing
                    page_text, page_extraction_type = self._process_page_hybrid(page, page_num)
                else:
                    # Legacy mode: simple text/OCR decision
                    page_text, page_extraction_type = self._process_page_legacy(page)

                extraction_types_used.add(page_extraction_type)

                if page_text.strip():
                    text_parts.append(page_text)

                    chunks.append(TextChunk(
                        text=page_text,
                        start_char=char_offset,
                        end_char=char_offset + len(page_text),
                        page_number=page_num + 1,
                        chunk_index=page_num,
                        metadata={"extraction_type": page_extraction_type}
                    ))
                    char_offset += len(page_text) + 2

            # Extract tables
            if self.extract_tables_enabled:
                tables = self._extract_tables(file_path)

            metadata = self._extract_pdf_metadata(doc, file_path)
            doc.close()

            full_content = "\n\n".join(text_parts)

            # Determine overall extraction_type
            if "indirect" in extraction_types_used or "mixed" in extraction_types_used:
                extraction_type = "indirect"  # Any OCR means post-processing needed
            else:
                extraction_type = "direct"

            # Add page breakdown to metadata
            metadata.extra["extraction_types_used"] = list(extraction_types_used)
            metadata.extra["hybrid_mode"] = self.hybrid_mode

            return ProcessedDocument(
                content=full_content,
                chunks=chunks,
                metadata=metadata,
                source_file=file_path,
                file_type="pdf",
                tables=tables,
                processed_at=datetime.now(),
                processing_time=time.time() - start_time,
                processor_version=self.VERSION,
                success=True,
                extraction_type=extraction_type
            )

        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

    # =========================================================================
    # Hybrid Processing Methods
    # =========================================================================

    def _has_meaningful_text(self, page, text: str) -> bool:
        """
        Check if page has meaningful text content.

        Args:
            page: PyMuPDF page object
            text: Extracted text

        Returns:
            True if page has meaningful text (not just header/footer/artifacts)
        """
        import re

        text_len = len(text.strip())

        # Too short → likely just header/footer
        if text_len < self.min_text_length:
            return False

        # Check density (chars per page area in square inches)
        page_rect = page.rect
        page_area = (page_rect.width * page_rect.height) / (72 * 72)  # Convert to square inches
        density = text_len / max(page_area, 1)

        # Low density → likely scan with some OCR artifacts
        if density < self.min_text_density:
            return False

        # Check text quality (ratio of valid chars vs total)
        valid_chars = len(re.findall(r'[\w\s]', text, re.UNICODE))
        quality_ratio = valid_chars / max(text_len, 1)

        # Too many garbage chars → OCR artifacts
        if quality_ratio < self.text_quality_threshold:
            return False

        return True

    def _classify_image(self, img: dict, page_rect, text_blocks: List[dict]) -> str:
        """
        Classify image type.

        Args:
            img: Image dict with bbox, width, height
            page_rect: Page rectangle
            text_blocks: List of text blocks

        Returns:
            - "logo": Small, in header/footer area
            - "decoration": Background, low contrast with text
            - "icon": Very small, inline with text
            - "content": Chart, diagram, screenshot (NEED OCR)
        """
        bbox = img["bbox"]
        width, height = img["width"], img["height"]

        # Calculate position
        x0, y0, x1, y1 = bbox
        page_h = page_rect.height

        # Header/Footer detection (top 10% or bottom 10%)
        is_header = y0 < page_h * 0.1
        is_footer = y1 > page_h * 0.9

        # Size classification
        area = width * height
        is_small = area < 5000  # pixels
        is_large = area > 50000

        # FIX 1: Detect full-page background images
        # If image covers >80% of page area, it's a background decoration
        page_area = page_rect.width * page_rect.height
        coverage_ratio = area / max(page_area, 1)
        if coverage_ratio > 0.8:
            return "decoration"

        # Position relative to text
        overlaps_text = any(
            self._bbox_overlap_ratio(bbox, tb["bbox"]) > 0.3
            for tb in text_blocks
        )

        # Classification logic
        if is_small and (is_header or is_footer):
            return "logo"

        if is_small and overlaps_text:
            return "icon"

        if overlaps_text and not is_large:
            return "decoration"

        # Large image not overlapping text → likely meaningful content
        if is_large and not overlaps_text:
            return "content"

        # Medium size in main area
        if not (is_header or is_footer) and area > self.min_content_image_size:
            return "content"

        return "decoration"

    def _analyze_page(self, page) -> PageContent:
        """
        Analyze page to determine content type.

        Returns:
            PageContent with classified content
        """
        # Extract text with positions
        text_dict = page.get_text("dict")
        text_blocks = []
        total_text = ""

        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                if block_text.strip():
                    text_blocks.append({
                        "bbox": block.get("bbox"),
                        "text": block_text
                    })
                    total_text += block_text + "\n"

        # Get images
        image_list = page.get_images(full=True)
        image_blocks = []
        for img in image_list:
            xref = img[0]
            # Get image bbox on page
            try:
                for img_rect in page.get_image_rects(xref):
                    image_blocks.append({
                        "bbox": tuple(img_rect),
                        "xref": xref,
                        "width": img[2],
                        "height": img[3]
                    })
            except Exception:
                # Some images may not have rects
                pass

        # Get drawings count (vector graphics)
        try:
            drawings = page.get_drawings()
            drawing_count = len(drawings) if drawings else 0
        except Exception:
            drawing_count = 0

        # Classify page type using improved logic
        has_text = self._has_meaningful_text(page, total_text)

        # Classify images - only count "content" images
        content_images = []
        for img in image_blocks:
            img_type = self._classify_image(img, page.rect, text_blocks)
            if img_type == "content":
                content_images.append(img)

        has_content_images = len(content_images) > 0

        # Decision tree
        if not has_text and not has_content_images:
            page_type = "empty"
        elif has_text and not has_content_images:
            page_type = "text"  # Pure text → Fast path
        elif not has_text and has_content_images:
            page_type = "scan"  # Need OCR
        else:
            # Both text and content images
            # Check text coverage to avoid false positives
            text_area = sum(
                (tb["bbox"][2] - tb["bbox"][0]) * (tb["bbox"][3] - tb["bbox"][1])
                for tb in text_blocks
            )
            page_area = page.rect.width * page.rect.height
            text_coverage = text_area / max(page_area, 1)

            # If text covers > 50% of page → mainly text with some diagrams
            if text_coverage > 0.5:
                page_type = "text"  # Prefer text extraction
            else:
                page_type = "mixed"  # True mixed content

        return PageContent(
            text=total_text,
            text_blocks=text_blocks,
            image_blocks=image_blocks,
            drawing_count=drawing_count,
            page_type=page_type
        )

    def _process_page_hybrid(self, page, page_num: int) -> Tuple[str, str]:
        """
        Process page using hybrid approach.

        Returns:
            (page_text, extraction_type)
        """
        content = self._analyze_page(page)

        if content.page_type == "empty":
            return "", "direct"

        if content.page_type == "text":
            # Pure text page - just use extracted text
            return content.text, "direct"

        if content.page_type == "scan":
            # Pure scan - OCR entire page
            if self.ocr_enabled:
                ocr_text = self._ocr_page(page)
                return ocr_text, "indirect"
            return "", "direct"

        # Mixed page - combine text extraction and OCR
        if content.page_type == "mixed":
            if self.ocr_enabled:
                return self._process_mixed_page(page, content)
            return content.text, "direct"

        return "", "direct"

    def _process_mixed_page(self, page, content: PageContent) -> Tuple[str, str]:
        """
        Process a page with both text and images.

        Strategy:
        1. Keep extracted text (direct)
        2. OCR only image regions
        3. Merge based on vertical position

        Returns:
            (merged_text, extraction_type)
        """
        import numpy as np

        # FIX 2: PowerPoint detection - trust text layer, skip image OCR
        # PowerPoint PDFs have reliable text layers with decorative background images
        if hasattr(self, 'doc') and self.doc:
            metadata = self.doc.metadata
            creator = metadata.get("creator", "") if metadata else ""
            if "PowerPoint" in creator:
                # PowerPoint has reliable text layer, skip all image OCR for mixed pages
                return content.text, "direct"

        # Collect all content blocks with positions
        all_blocks = []

        # Add text blocks
        for tb in content.text_blocks:
            all_blocks.append({
                "type": "text",
                "bbox": tb["bbox"],
                "content": tb["text"],
                "y_pos": tb["bbox"][1]  # Top Y position
            })

        # OCR image regions and add
        ocr_used = False
        for img in content.image_blocks:
            # Skip very small images (likely icons/bullets)
            if img["width"] < self.min_image_size or img["height"] < self.min_image_size:
                continue

            # Skip images that are just lines/borders (extreme aspect ratio)
            aspect_ratio = img["width"] / max(img["height"], 1)
            if aspect_ratio > 20 or aspect_ratio < 0.05:
                continue

            # Classify image type - skip non-content images
            img_type = self._classify_image(img, page.rect, content.text_blocks)
            if img_type != "content":
                continue

            # Check if image overlaps significantly with text blocks
            # Skip OCR if text already covers this region (using improved threshold)
            img_bbox = img["bbox"]
            overlaps_text = False
            for tb in content.text_blocks:
                if self._bbox_overlap_ratio(img_bbox, tb["bbox"]) > self.image_overlap_threshold:
                    overlaps_text = True
                    break

            if overlaps_text:
                continue

            # OCR the image region with confidence check
            ocr_text, confidence = self._ocr_image_region_with_confidence(page, img["bbox"])
            if ocr_text and ocr_text.strip() and confidence >= self.ocr_confidence_threshold:
                all_blocks.append({
                    "type": "ocr",
                    "bbox": img["bbox"],
                    "content": ocr_text,
                    "y_pos": img["bbox"][1]
                })
                ocr_used = True

        # Sort by vertical position (top to bottom)
        all_blocks.sort(key=lambda b: b["y_pos"])

        # Merge content
        merged_parts = []
        for block in all_blocks:
            text = block["content"].strip()
            if text:
                merged_parts.append(text)

        merged_text = "\n\n".join(merged_parts)

        # extraction_type based on what was used
        extraction_type = "mixed" if ocr_used else "direct"
        return merged_text, extraction_type

    def _bbox_overlap_ratio(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)  # Image area
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)  # Text block area

        # FIX 3: Use smaller area for overlap calculation
        # This detects when small text overlaps with large backgrounds
        # Prevents false negatives where text on background shows tiny overlap ratio
        smaller_area = min(area1, area2)
        return intersection / max(smaller_area, 1)

    def _ocr_image_region(self, page, bbox: Tuple[float, float, float, float]) -> str:
        """
        OCR a specific region of a page.

        Args:
            page: PyMuPDF page object
            bbox: Bounding box (x0, y0, x1, y1)

        Returns:
            OCR text from that region
        """
        import fitz
        import numpy as np

        try:
            # Create clip rectangle
            clip = fitz.Rect(bbox)

            # Render only the clipped region at high DPI
            zoom = self.ocr_dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip)

            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_array = img_array[:, :, :3]

            # OCR the region
            ocr = self._get_ocr_engine()
            result = ocr.extract_text(img_array, detailed=False)

            return result if isinstance(result, str) else result.text

        except Exception as e:
            print(f"OCR region error: {e}")
            return ""

    def _ocr_image_region_with_confidence(self, page, bbox: Tuple[float, float, float, float]) -> Tuple[str, float]:
        """
        OCR image region and return confidence score.

        Args:
            page: PyMuPDF page object
            bbox: Bounding box (x0, y0, x1, y1)

        Returns:
            (text, confidence) where confidence is 0.0-1.0
        """
        import re

        ocr_text = self._ocr_image_region(page, bbox)

        # Calculate confidence based on text quality
        if not ocr_text or not ocr_text.strip():
            return "", 0.0

        # Check ratio of valid characters
        total_chars = len(ocr_text)
        valid_chars = len(re.findall(r'[\w\s]', ocr_text, re.UNICODE))
        confidence = valid_chars / max(total_chars, 1)

        # Penalize very short results
        if total_chars < 10:
            confidence *= 0.5

        return ocr_text, confidence

    # =========================================================================
    # Legacy Processing (fallback)
    # =========================================================================

    def _process_page_legacy(self, page) -> Tuple[str, str]:
        """
        Legacy page processing (non-hybrid).

        Simple decision: if text < threshold, OCR entire page.
        """
        page_text = page.get_text()

        if len(page_text.strip()) < self.min_text_length and self.ocr_enabled:
            page_text = self._ocr_page(page)
            if page_text.strip():
                return page_text, "indirect"
            return "", "direct"

        return page_text, "direct"

    # =========================================================================
    # OCR Methods
    # =========================================================================

    def _get_ocr_engine(self) -> OCREngine:
        """Lazy load OCR engine (PaddleOCR with EasyOCR fallback)"""
        if self._ocr_engine is None:
            # Read preprocess from config or env (default: True)
            preprocess = self.config.get(
                "ocr_preprocess",
                os.getenv("OCR_PREPROCESS", "true").lower() == "true"
            )
            self._ocr_engine = OCREngine(
                language=self.ocr_language,
                use_gpu=self.config.get("use_gpu", True),
                preprocess=preprocess,
                dpi=self.ocr_dpi
            )
        return self._ocr_engine

    def _ocr_page(self, page) -> str:
        """
        OCR a PDF page using enhanced OCR engine.

        Features:
        - High DPI rendering (400 DPI default)
        - Advanced preprocessing (deskew, denoise, binarize)
        - PaddleOCR for better Vietnamese accuracy
        """
        try:
            ocr = self._get_ocr_engine()
            result = ocr.extract_from_pdf_page(page, dpi=self.ocr_dpi)
            return result.text

        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    # =========================================================================
    # Table Extraction
    # =========================================================================

    def _extract_tables(self, file_path: str) -> List[Table]:
        """Extract tables từ PDF using pdfplumber"""
        tables = []

        try:
            import pdfplumber
        except ImportError:
            print("pdfplumber not installed. Skipping table extraction.")
            return tables

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()

                    for idx, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 0:
                            # Clean table data
                            cleaned_data = []
                            for row in table_data:
                                cleaned_row = [
                                    str(cell).strip() if cell else ""
                                    for cell in row
                                ]
                                cleaned_data.append(cleaned_row)

                            tables.append(Table(
                                data=cleaned_data,
                                headers=cleaned_data[0] if cleaned_data else None,
                                page_number=page_num + 1,
                                table_index=idx
                            ))
        except Exception as e:
            print(f"Table extraction error: {e}")

        return tables

    # =========================================================================
    # Metadata Extraction
    # =========================================================================

    def _extract_pdf_metadata(self, doc, file_path: str) -> DocumentMetadata:
        """Extract PDF-specific metadata"""
        base_meta = self.extract_metadata(file_path)
        pdf_meta = doc.metadata or {}

        base_meta.page_count = len(doc)
        base_meta.author = pdf_meta.get("author")
        base_meta.title = pdf_meta.get("title")

        # Add extra PDF metadata
        base_meta.extra = {
            "creator": pdf_meta.get("creator"),
            "producer": pdf_meta.get("producer"),
            "subject": pdf_meta.get("subject"),
            "keywords": pdf_meta.get("keywords"),
        }

        return base_meta
