"""
Enhanced OCR Engine - PaddleOCR with advanced preprocessing.

Features:
- PaddleOCR (primary) - Best for Vietnamese + English
- EasyOCR (fallback) - Good balance
- Advanced image preprocessing for scanned documents
- High DPI rendering for PDF pages

Usage:
    from src.modules.document_processor.ocr_engine import OCREngine

    # Default: PaddleOCR with Vietnamese
    ocr = OCREngine()
    text = ocr.extract_text("image.png")

    # For PDF pages
    result = ocr.extract_from_pdf_page(page, dpi=300)
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load config from environment
from dotenv import load_dotenv
load_dotenv()

# Note: Text normalization is done centrally in knowledge_base.py
# Do NOT normalize here to avoid double-normalization


@dataclass
class OCRResult:
    """Single OCR detection result"""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


@dataclass
class OCROutput:
    """Full OCR output"""
    text: str
    results: List[OCRResult]
    processing_time: float
    engine: str
    confidence_avg: float


class ImagePreprocessor:
    """Advanced image preprocessing for better OCR accuracy"""

    @staticmethod
    def preprocess(
        image,
        min_dimension: int = 1500,
        deskew: bool = True,
        denoise: bool = True,
        binarize: bool = True,
        sharpen: bool = False
    ) -> np.ndarray:
        """
        Preprocess image for OCR.

        Args:
            image: PIL Image, numpy array, or file path
            min_dimension: Minimum image dimension (upscale if smaller)
            deskew: Fix tilted scans
            denoise: Remove noise
            binarize: Convert to binary (black/white)
            sharpen: Apply sharpening

        Returns:
            Preprocessed image as numpy array
        """
        try:
            import cv2
            from PIL import Image
        except ImportError as e:
            print(f"Warning: Preprocessing requires opencv-python and Pillow: {e}")
            if isinstance(image, (str, Path)):
                return np.array(Image.open(image))
            return np.array(image) if not isinstance(image, np.ndarray) else image

        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            img_array = np.array(img)
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image)

        # Ensure RGB/BGR format
        if len(img_array.shape) == 2:
            # Grayscale, convert to BGR for consistency
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            # RGBA, convert to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        # 1. Upscale if too small
        if min(img_array.shape[:2]) < min_dimension:
            scale = min_dimension / min(img_array.shape[:2])
            new_w = int(img_array.shape[1] * scale)
            new_h = int(img_array.shape[0] * scale)
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale for processing
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # 3. Deskew (fix rotation)
        if deskew:
            gray = ImagePreprocessor._deskew(gray)

        # 4. Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)

        # 5. Binarization (adaptive threshold)
        if binarize:
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

        # 6. Morphological cleanup
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # 7. Sharpen (optional)
        if sharpen:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel)

        return gray

    @staticmethod
    def _deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """Fix tilted image"""
        try:
            import cv2

            # Find contours to detect text orientation
            coords = np.column_stack(np.where(image > 0))

            if len(coords) < 100:
                return image

            # Get minimum area rectangle
            angle = cv2.minAreaRect(coords)[-1]

            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90

            # Only deskew if angle is significant but not too extreme
            if 0.5 < abs(angle) < max_angle:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        except Exception:
            pass

        return image

    @staticmethod
    def preprocess_rgb(
        image,
        min_dimension: int = 2000,
        denoise: bool = True,
        sharpen: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for PaddleOCR (keeps RGB format).

        PaddleOCR 3.x requires RGB input, so we can't convert to grayscale.
        This method applies enhancements while keeping color.

        Args:
            image: PIL Image, numpy array, or file path
            min_dimension: Minimum image dimension (upscale if smaller)
            denoise: Remove noise (color-preserving)
            sharpen: Apply sharpening for clearer text

        Returns:
            Preprocessed RGB image as numpy array
        """
        try:
            import cv2
            from PIL import Image
        except ImportError as e:
            print(f"Warning: Preprocessing requires opencv-python and Pillow: {e}")
            if isinstance(image, (str, Path)):
                return np.array(Image.open(image))
            return np.array(image) if not isinstance(image, np.ndarray) else image

        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            img_array = np.array(img)
        elif isinstance(image, np.ndarray):
            img_array = image.copy()
        else:
            img_array = np.array(image)

        # Ensure RGB format (convert grayscale to RGB)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] == 3:
            # Check if BGR (from cv2) and convert to RGB
            # Assume it's already RGB from PIL
            pass

        # 1. Upscale if too small (important for OCR accuracy)
        if min(img_array.shape[:2]) < min_dimension:
            scale = min_dimension / min(img_array.shape[:2])
            new_w = int(img_array.shape[1] * scale)
            new_h = int(img_array.shape[0] * scale)
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 2. Denoise (color-preserving)
        if denoise:
            # Use color denoising for RGB images
            img_array = cv2.fastNlMeansDenoisingColored(img_array, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)

        # 3. Sharpen for clearer text edges
        if sharpen:
            # Unsharp masking
            gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
            img_array = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)

        # 4. Enhance contrast
        # Convert to LAB and enhance L channel
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return img_array

    @staticmethod
    def preprocess_pdf_page(page_pixmap, target_dpi: int = 400) -> np.ndarray:
        """
        Preprocess PDF page pixmap for OCR.

        Args:
            page_pixmap: PyMuPDF pixmap object
            target_dpi: Target DPI (300 recommended for OCR)

        Returns:
            Preprocessed image as numpy array
        """
        # Convert pixmap to numpy array
        img_array = np.frombuffer(page_pixmap.samples, dtype=np.uint8)
        img_array = img_array.reshape(page_pixmap.height, page_pixmap.width, page_pixmap.n)

        # Remove alpha channel if present
        if page_pixmap.n == 4:
            img_array = img_array[:, :, :3]

        # Apply preprocessing with settings optimized for PDF
        return ImagePreprocessor.preprocess(
            img_array,
            min_dimension=2000,  # Higher for PDF
            deskew=True,
            denoise=True,
            binarize=True,
            sharpen=False  # PDFs usually don't need sharpening
        )


class OCREngine:
    """
    Main OCR Engine using PaddleOCR with EasyOCR fallback.

    Configuration via .env:
        OCR_ENGINE=paddleocr  # paddleocr, easyocr
        OCR_LANGUAGE=vi       # vi, en, ch
        OCR_USE_GPU=true
        OCR_DPI=400           # Higher DPI for better Vietnamese OCR
        OCR_MIN_CONFIDENCE=0.5
        OCR_PREPROCESS=true
    """

    def __init__(
        self,
        engine: str = None,
        language: str = None,
        use_gpu: bool = None,
        preprocess: bool = None,
        min_confidence: float = None,
        dpi: int = None
    ):
        """
        Initialize OCR Engine.

        Args:
            engine: "paddleocr" (default) or "easyocr"
            language: "vi" (default), "en", "ch"
            use_gpu: Use GPU acceleration (default: from env or True)
            preprocess: Enable preprocessing (default: True)
            min_confidence: Minimum confidence threshold (default: 0.5)
            dpi: DPI for PDF rendering (default: 400)
        """
        # Load from env with fallbacks
        self.engine_name = engine or os.getenv("OCR_ENGINE", "paddleocr")
        self.language = language or os.getenv("OCR_LANGUAGE", "vi")
        self.use_gpu = use_gpu if use_gpu is not None else os.getenv("OCR_USE_GPU", "true").lower() == "true"
        self.preprocess_enabled = preprocess if preprocess is not None else os.getenv("OCR_PREPROCESS", "true").lower() == "true"
        self.min_confidence = min_confidence or float(os.getenv("OCR_MIN_CONFIDENCE", "0.5"))
        self.dpi = dpi or int(os.getenv("OCR_DPI", "400"))  # Higher DPI for better Vietnamese

        # Image size limit to prevent crashes (PaddleOCR limit ~4000px)
        self.max_image_size = int(os.getenv("OCR_MAX_IMAGE_SIZE", "3500"))

        self._reader = None
        self._initialized = False

    def _resize_if_needed(self, img_array: np.ndarray) -> np.ndarray:
        """
        Resize image if it exceeds max_image_size to prevent OCR crashes.

        Args:
            img_array: Input image as numpy array

        Returns:
            Resized image if needed, otherwise original
        """
        h, w = img_array.shape[:2]
        max_dim = max(h, w)

        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)

            try:
                import cv2
                resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"[OCR] Resized image from {w}x{h} to {new_w}x{new_h} (max={self.max_image_size})")
                return resized
            except ImportError:
                from PIL import Image
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                print(f"[OCR] Resized image from {w}x{h} to {new_w}x{new_h} (max={self.max_image_size})")
                return np.array(pil_img)

        return img_array

    def _initialize(self):
        """Lazy initialization of OCR reader"""
        if self._initialized:
            return

        # Fix Windows asyncio issue
        if sys.platform == "win32":
            import asyncio
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

        # Try PaddleOCR first
        if self.engine_name == "paddleocr":
            try:
                self._init_paddleocr()
                self._initialized = True
                print(f"[OCR] Initialized PaddleOCR (lang={self.language}, gpu={self.use_gpu})")
                return
            except Exception as e:
                print(f"[OCR] PaddleOCR failed: {e}, falling back to EasyOCR")
                self.engine_name = "easyocr"

        # Fallback to EasyOCR
        if self.engine_name == "easyocr":
            try:
                self._init_easyocr()
                self._initialized = True
                print(f"[OCR] Initialized EasyOCR (lang={self.language}, gpu={self.use_gpu})")
                return
            except Exception as e:
                raise RuntimeError(
                    f"No OCR engine available. Install PaddleOCR:\n"
                    f"  pip install paddleocr paddlepaddle\n"
                    f"Or EasyOCR:\n"
                    f"  pip install easyocr\n"
                    f"Error: {e}"
                )

    def _init_paddleocr(self):
        """Initialize PaddleOCR (v3.x API)"""
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x uses different parameters than 2.x
        # Disable optional models to reduce memory usage:
        # - use_doc_orientation_classify: for rotated documents (not needed for scanned PDFs)
        # - use_doc_unwarping: for curved/warped documents (not needed for flat scans)
        # - use_textline_orientation: for vertical text (Vietnamese is horizontal)
        self._reader = PaddleOCR(
            lang=self.language,
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            use_doc_orientation_classify=False,  # Disable to save ~200MB
            use_doc_unwarping=False,             # Disable to save ~300MB
            use_textline_orientation=False,      # Disable to save ~200MB
        )

    def _init_easyocr(self):
        """Initialize EasyOCR"""
        import easyocr

        # Map language for EasyOCR (supports multiple)
        langs = [self.language]
        if self.language != "en":
            langs.append("en")

        self._reader = easyocr.Reader(
            langs,
            gpu=self.use_gpu,
            verbose=False
        )

    def extract_text(self, image, detailed: bool = False) -> Union[str, OCROutput]:
        """
        Extract text from image.

        Args:
            image: File path, PIL Image, numpy array, or bytes
            detailed: Return full OCROutput instead of just text

        Returns:
            Extracted text, or OCROutput if detailed=True
        """
        self._initialize()
        start_time = time.time()

        # Load image
        if isinstance(image, (str, Path)):
            from PIL import Image as PILImage
            img_array = np.array(PILImage.open(image))
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image)

        # Resize if too large (prevent crashes)
        img_array = self._resize_if_needed(img_array)

        # Run OCR with appropriate preprocessing
        if self.engine_name == "paddleocr":
            # PaddleOCR 3.x requires RGB input
            # Use RGB preprocessing (upscale, denoise, sharpen, contrast)
            if self.preprocess_enabled:
                processed = ImagePreprocessor.preprocess_rgb(img_array)
            else:
                processed = img_array
            results = self._ocr_paddle(processed)
        else:
            # EasyOCR benefits from grayscale preprocessing
            if self.preprocess_enabled:
                processed = ImagePreprocessor.preprocess(img_array)
            else:
                processed = img_array
            results = self._ocr_easyocr(processed)

        # Build output
        full_text = "\n".join([r.text for r in results])
        avg_conf = sum(r.confidence for r in results) / len(results) if results else 0

        output = OCROutput(
            text=full_text,
            results=results,
            processing_time=time.time() - start_time,
            engine=self.engine_name,
            confidence_avg=avg_conf
        )

        return output if detailed else full_text

    def _ocr_paddle(self, image: np.ndarray) -> List[OCRResult]:
        """Run PaddleOCR (v3.x API)"""
        # PaddleOCR 3.x uses predict() instead of ocr()
        results = self._reader.predict(image)
        ocr_results = []

        if results and len(results) > 0:
            result = results[0]  # First image result

            # PaddleOCR 3.x output format:
            # - rec_texts: list of recognized texts
            # - rec_scores: list of confidence scores
            # - rec_polys: list of polygon coordinates
            rec_texts = result.get('rec_texts', [])
            rec_scores = result.get('rec_scores', [])
            rec_polys = result.get('rec_polys', [])

            for i, (text, conf) in enumerate(zip(rec_texts, rec_scores)):
                if conf >= self.min_confidence:
                    # Get bbox from polygon if available
                    bbox = None
                    if i < len(rec_polys) and len(rec_polys[i]) > 0:
                        poly = rec_polys[i]
                        x_coords = [p[0] for p in poly]
                        y_coords = [p[1] for p in poly]
                        bbox = (
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords))
                        )

                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=float(conf),
                        bbox=bbox
                    ))

        return ocr_results

    def _ocr_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Run EasyOCR"""
        results = self._reader.readtext(image)
        ocr_results = []

        for bbox_points, text, conf in results:
            if conf >= self.min_confidence:
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                bbox = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                )

                ocr_results.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=bbox
                ))

        return ocr_results

    def extract_from_pdf_page(self, page, dpi: int = None) -> OCROutput:
        """
        Extract text from PDF page using OCR.

        Args:
            page: PyMuPDF page object
            dpi: Rendering DPI (default: self.dpi or 400)

        Returns:
            OCROutput with extracted text
        """
        import fitz

        self._initialize()
        start_time = time.time()
        dpi = dpi or self.dpi

        # Render page at high resolution
        zoom = dpi / 72  # 72 is default PDF DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to numpy array
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_array = img_array[:, :, :3]

        # Resize if too large (prevent crashes)
        img_array = self._resize_if_needed(img_array)

        # Run OCR with appropriate preprocessing
        if self.engine_name == "paddleocr":
            # PaddleOCR 3.x requires RGB - use RGB preprocessing
            if self.preprocess_enabled:
                processed = ImagePreprocessor.preprocess_rgb(img_array)
            else:
                processed = img_array
            results = self._ocr_paddle(processed)
        else:
            # EasyOCR benefits from grayscale preprocessing
            if self.preprocess_enabled:
                processed = ImagePreprocessor.preprocess(img_array)
            else:
                processed = img_array
            results = self._ocr_easyocr(processed)

        # Build output
        full_text = "\n".join([r.text for r in results])
        avg_conf = sum(r.confidence for r in results) / len(results) if results else 0

        return OCROutput(
            text=full_text,
            results=results,
            processing_time=time.time() - start_time,
            engine=self.engine_name,
            confidence_avg=avg_conf
        )

    @staticmethod
    def available_engines() -> List[str]:
        """List available OCR engines"""
        available = []

        try:
            import paddleocr
            available.append("paddleocr")
        except ImportError:
            pass

        try:
            import easyocr
            available.append("easyocr")
        except ImportError:
            pass

        return available


# Convenience function
def ocr_image(image, language: str = "vi", preprocess: bool = True) -> str:
    """
    Quick OCR function.

    Args:
        image: Image file path, PIL Image, or numpy array
        language: Language code ("vi", "en", "ch")
        preprocess: Enable preprocessing

    Returns:
        Extracted text
    """
    ocr = OCREngine(language=language, preprocess=preprocess)
    return ocr.extract_text(image)
