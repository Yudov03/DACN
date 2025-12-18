# -*- coding: utf-8 -*-
"""
Post-Processing Module
======================

Post-process extracted text to fix OCR/ASR errors.

Text Types:
    - DIRECT: Extracted directly from documents (doc, txt, md, code, etc.)
    - INDIRECT: Extracted via OCR (pdf, images) or ASR (audio, video)

Processing Methods:
    - none: No processing
    - transformer: bmd1905/vietnamese-correction-v2 (HuggingFace)
    - ollama: qwen2.5:7b with unified Vietnamese correction prompt

Usage:
    from src.modules.post_processing import PostProcessor, post_process_text

    # Auto-detect and process
    processor = PostProcessor()
    result = processor.process(text, extraction_type="indirect")

    # Quick function
    result = post_process_text(text, extraction_type="indirect")

Configuration (.env):
    POSTPROCESS_DIRECT=none           # none, transformer, ollama
    POSTPROCESS_INDIRECT=ollama       # none, transformer, ollama
    POSTPROCESS_OLLAMA_MODEL=qwen2.5:7b
    POSTPROCESS_TRANSFORMER_MODEL=bmd1905/vietnamese-correction-v2
"""

import os
import sys
import time
import shutil
import logging
import subprocess
import requests
from typing import Literal, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Ollama Auto-Start
# =============================================================================

def start_ollama_server(timeout: int = 30) -> bool:
    """
    Start Ollama server if not running.

    Args:
        timeout: Max seconds to wait for server to start

    Returns:
        True if server is running (started or already running)
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Check if already running
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("Ollama server already running")
            return True
    except Exception:
        pass

    # Find ollama executable
    ollama_cmd = shutil.which("ollama")
    if not ollama_cmd:
        # Try common Windows paths
        common_paths = [
            r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.getenv("USERNAME", "")),
            r"C:\Program Files\Ollama\ollama.exe",
            r"C:\Program Files (x86)\Ollama\ollama.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                ollama_cmd = path
                break

    if not ollama_cmd:
        logger.warning("Ollama not found. Install from https://ollama.com/download")
        return False

    logger.info(f"Starting Ollama server: {ollama_cmd}")

    try:
        # Start ollama serve in background
        if sys.platform == "win32":
            # Windows: use CREATE_NO_WINDOW flag
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                [ollama_cmd, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            )
        else:
            # Linux/Mac
            process = subprocess.Popen(
                [ollama_cmd, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{base_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    logger.info("Ollama server started successfully")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        logger.warning(f"Ollama server did not start within {timeout}s")
        return False

    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False


# =============================================================================
# Types
# =============================================================================

ExtractionType = Literal["direct", "indirect"]
ProcessingMethod = Literal["none", "transformer", "ollama"]


# =============================================================================
# Cache
# =============================================================================

import hashlib
import json
from pathlib import Path
from datetime import datetime


class PostProcessingCache:
    """
    Cache for post-processing results.

    Stores processed text by hash of (input_text + method + model).
    Avoids re-processing same text when re-indexing documents.

    Usage:
        cache = PostProcessingCache()
        cached = cache.get(text, "ollama", "qwen2.5:7b")
        if cached is None:
            result = process_with_ollama(text)
            cache.set(text, "ollama", "qwen2.5:7b", result)
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files.
                       Default: data/cache/post_processing
        """
        self.cache_dir = Path(cache_dir or os.getenv(
            "POSTPROCESS_CACHE_DIR",
            "data/cache/post_processing"
        ))
        self.chunks_dir = self.cache_dir / "chunks"
        self.index_file = self.cache_dir / "index.json"
        self._index = None
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create cache directories if not exist."""
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict:
        """Load or create index."""
        if self._index is None:
            if self.index_file.exists():
                try:
                    with open(self.index_file, 'r', encoding='utf-8') as f:
                        self._index = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._index = {}
            else:
                self._index = {}
        return self._index

    def _save_index(self):
        """Save index to file."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache index: {e}")

    def get_cache_key(self, text: str, method: str, model: str) -> str:
        """
        Generate unique cache key.

        Key is MD5 hash of: text + method + model
        This ensures different models/methods don't share cache.
        """
        content = f"{text}|{method}|{model}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(self, text: str, method: str, model: str) -> Optional[str]:
        """
        Get cached result if exists.

        Args:
            text: Original input text
            method: Processing method (ollama, transformer)
            model: Model name

        Returns:
            Cached result or None if not found
        """
        key = self.get_cache_key(text, method, model)
        index = self._load_index()

        if key not in index:
            return None

        cache_file = self.chunks_dir / f"{key}.txt"
        if not cache_file.exists():
            # Index out of sync, remove entry
            del index[key]
            self._index = index
            self._save_index()
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError:
            return None

    def set(self, text: str, method: str, model: str, result: str):
        """
        Cache a result.

        Args:
            text: Original input text
            method: Processing method
            model: Model name
            result: Processed result to cache
        """
        key = self.get_cache_key(text, method, model)

        # Save result to file
        cache_file = self.chunks_dir / f"{key}.txt"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(result)
        except IOError as e:
            logger.warning(f"Failed to save cache file: {e}")
            return

        # Update index
        index = self._load_index()
        index[key] = {
            "method": method,
            "model": model,
            "input_length": len(text),
            "output_length": len(result),
            "created_at": datetime.now().isoformat()
        }
        self._index = index
        self._save_index()

    def clear(self):
        """Clear all cache."""
        import shutil as shutil_module
        if self.cache_dir.exists():
            shutil_module.rmtree(self.cache_dir)
        self._index = None
        self._ensure_dirs()
        logger.info("Post-processing cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        index = self._load_index()

        total_size = 0
        valid_entries = 0
        for key in index:
            cache_file = self.chunks_dir / f"{key}.txt"
            if cache_file.exists():
                total_size += cache_file.stat().st_size
                valid_entries += 1

        return {
            "entries": valid_entries,
            "total_entries": len(index),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }


# Global cache instance (lazy loaded)
_cache_instance: Optional[PostProcessingCache] = None


def get_cache() -> Optional[PostProcessingCache]:
    """Get global cache instance if caching is enabled."""
    global _cache_instance

    if os.getenv("POSTPROCESS_CACHE", "true").lower() != "true":
        return None

    if _cache_instance is None:
        _cache_instance = PostProcessingCache()

    return _cache_instance


# =============================================================================
# Constants
# =============================================================================

# Vietnamese correction prompt - strict output only
VIETNAMESE_CORRECTION_PROMPT = """[TASK] Sửa lỗi chính tả tiếng Việt. Thêm dấu thanh bị thiếu.
[RULE] CHỈ trả về văn bản đã sửa. KHÔNG giải thích. KHÔNG thêm gì khác.
[INPUT]
{text}
[OUTPUT]"""

# Module-level state to avoid reloading model
_ollama_model_loaded = False
_ollama_last_model = None


# =============================================================================
# Transformer Processor
# =============================================================================

class TransformerProcessor:
    """Vietnamese text correction using HuggingFace Transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "POSTPROCESS_TRANSFORMER_MODEL",
            "bmd1905/vietnamese-correction-v2"
        )
        self._pipeline = None
        logger.info(f"TransformerProcessor initialized with model: {self.model_name}")

    @property
    def pipeline(self):
        """Lazy load transformer pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading transformer model: {self.model_name}")
                self._pipeline = pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    device=-1  # CPU, use 0 for GPU
                )
                logger.info("Transformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load transformer model: {e}")
                raise
        return self._pipeline

    def process(self, text: str) -> str:
        """Process text using transformer model."""
        if not text or not text.strip():
            return text

        # Check cache first
        cache = get_cache()
        if cache:
            cached_result = cache.get(text, "transformer", self.model_name)
            if cached_result is not None:
                logger.info(f"Transformer cache HIT: {len(text)} -> {len(cached_result)} chars")
                return cached_result

        try:
            # Process in chunks if text is too long
            max_length = 512
            if len(text) <= max_length:
                result = self.pipeline(text, max_length=max_length, num_beams=5)
                output = result[0]["generated_text"]
                # Save to cache
                if cache:
                    cache.set(text, "transformer", self.model_name, output)
                return output

            # Split by lines and process each
            lines = text.split("\n")
            processed_lines = []

            for line in lines:
                if not line.strip():
                    processed_lines.append(line)
                elif len(line) <= max_length:
                    result = self.pipeline(line, max_length=max_length, num_beams=5)
                    processed_lines.append(result[0]["generated_text"])
                else:
                    # Line too long, keep as is
                    processed_lines.append(line)

            final_result = "\n".join(processed_lines)

            # Save to cache
            if cache:
                cache.set(text, "transformer", self.model_name, final_result)

            return final_result

        except Exception as e:
            logger.error(f"Transformer processing error: {e}")
            return text

    def is_available(self) -> bool:
        """Check if transformer is available."""
        try:
            _ = self.pipeline
            return True
        except Exception:
            return False


# =============================================================================
# Ollama Processor
# =============================================================================

class OllamaProcessor:
    """Vietnamese text correction using Ollama LLM."""

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        timeout: int = None,  # None = no timeout
        auto_start: bool = None
    ):
        self.model = model or os.getenv("POSTPROCESS_OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout  # None means no timeout
        self.auto_start = auto_start if auto_start is not None else \
            os.getenv("OLLAMA_AUTO_START", "true").lower() == "true"

        # State tracking (use module-level to persist across instances)
        self._server_available = None
        self._last_check_time = 0

        logger.info(f"OllamaProcessor: model={self.model}, timeout={timeout}")

    def _check_server(self, force: bool = False) -> bool:
        """
        Check if Ollama server is available.

        Args:
            force: Force recheck even if recently checked
        """
        # Rate limit checks (max once per 30 seconds unless forced)
        now = time.time()
        if not force and self._server_available is not None:
            if now - self._last_check_time < 30:
                return self._server_available

        self._last_check_time = now

        # Check if running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self._server_available = True
                logger.info("Ollama server is available")
                return True
        except Exception as e:
            logger.warning(f"Ollama server check failed: {e}")

        self._server_available = False

        # Auto-start if not running
        if self.auto_start:
            logger.info("Attempting to auto-start Ollama...")
            if start_ollama_server(timeout=30):
                self._server_available = True
                return True

        return False

    def _ensure_model_loaded(self) -> bool:
        """Ensure model is loaded in memory (warm up)."""
        global _ollama_model_loaded, _ollama_last_model

        # Already loaded for this model
        if _ollama_model_loaded and _ollama_last_model == self.model:
            print(f"      [Ollama] Model '{self.model}' already loaded", flush=True)
            return True

        if not self._check_server():
            print(f"      [Ollama] Server not available at {self.base_url}", flush=True)
            return False

        print(f"      [Ollama] Loading model '{self.model}'...", flush=True, end=" ")
        start = time.time()

        try:
            # Simple request to load model with SAME num_ctx as actual requests
            # This prevents reloading when first real request comes
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "test",
                    "stream": False,
                    "keep_alive": "30m",  # Keep model loaded
                    "options": {
                        "num_predict": 1,
                        "num_ctx": 4096  # Same as actual requests
                    }
                },
                timeout=300  # 5 minutes for cold start, then cached
            )

            elapsed = time.time() - start

            if response.status_code == 200:
                _ollama_model_loaded = True
                _ollama_last_model = self.model
                print(f"OK ({elapsed:.1f}s)", flush=True)
                return True
            else:
                print(f"FAILED (HTTP {response.status_code})", flush=True)
                return False

        except requests.Timeout:
            print(f"TIMEOUT after 300s", flush=True)
            return False
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            return False

    def _call_ollama(self, text: str) -> str:
        """
        Call Ollama API for text correction.

        Single request, no retry - caller handles retry if needed.
        """
        global _ollama_model_loaded  # Must declare at top of function

        prompt = VIETNAMESE_CORRECTION_PROMPT.format(text=text)

        # Calculate reasonable num_predict
        # Vietnamese: ~1.5 chars per token on average
        estimated_tokens = len(text) // 2 + 100  # Add buffer
        num_predict = min(estimated_tokens, 4096)  # Cap at 4096

        # FIXED context size to prevent model reloading
        # Ollama reloads model when num_ctx changes!
        num_ctx = 4096  # Fixed - prevents model reload between requests

        logger.info(f"Ollama request: {len(text)} chars, num_predict={num_predict}")

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",  # Keep model loaded for 30 minutes
                    "options": {
                        "temperature": 0.1,
                        "num_predict": num_predict,
                        "num_ctx": num_ctx,
                    }
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    logger.info(f"Ollama response: {len(result)} chars")
                    return result
                else:
                    logger.warning("Ollama returned empty response")
            else:
                logger.error(f"Ollama API error: HTTP {response.status_code}")
                # Reset model loaded flag on 500 error
                if response.status_code == 500:
                    _ollama_model_loaded = False

        except requests.Timeout:
            logger.error(f"Ollama request timed out ({self.timeout}s)") if self.timeout else None
        except Exception as e:
            logger.error(f"Ollama request error: {e}")

        return None  # Return None on failure, caller decides what to do

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences at sentence boundaries."""
        import re

        # First, normalize newlines and split by them to preserve structure
        lines = text.split('\n')

        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split line by sentence-ending punctuation
            # This regex splits after . ! ? but keeps the punctuation
            parts = re.split(r'(?<=[.!?])\s+', line)
            sentences.extend([p.strip() for p in parts if p.strip()])

        return sentences

    def _create_chunks_by_sentence(self, text: str, max_chars: int) -> list:
        """
        Split text into chunks at sentence boundaries.
        Each chunk ends at a complete sentence (period, !, ?).
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence) + 1  # +1 for space/newline

            # If single sentence is too long, it becomes its own chunk
            if sent_len > max_chars:
                # Save current chunk first
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                # Add long sentence as its own chunk
                chunks.append(sentence)
                continue

            # If adding this sentence exceeds limit, start new chunk
            if current_len + sent_len > max_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_len = 0

            current_chunk.append(sentence)
            current_len += sent_len

        # Don't forget last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process(self, text: str, max_chunk_chars: int = 1500) -> str:
        """
        Process text with Ollama.

        Args:
            text: Input text to process
            max_chunk_chars: Max chars per chunk (default 1500 ~ 750 tokens)

        Returns:
            Processed text (or original if processing fails)
        """
        if not text or not text.strip():
            return text

        # Check cache first
        cache = get_cache()
        if cache:
            cached_result = cache.get(text, "ollama", self.model)
            if cached_result is not None:
                print(f"      [Ollama] Cache HIT ({len(text):,} chars -> {len(cached_result):,} chars)", flush=True)
                return cached_result

        # Ensure server and model are ready
        if not self._ensure_model_loaded():
            print(f"      [Ollama] Server not available, skipping", flush=True)
            return text

        # Small file: process as single request
        if len(text) <= max_chunk_chars:
            print(f"      [Ollama] Single request ({len(text):,} chars, no chunking)", flush=True)
            start = time.time()
            result = self._call_ollama(text)
            elapsed = time.time() - start
            if result:
                print(f"      [Ollama] Done in {elapsed:.1f}s -> {len(result):,} chars", flush=True)
                # Save to cache
                if cache:
                    cache.set(text, "ollama", self.model, result)
                return result
            else:
                print(f"      [Ollama] Failed after {elapsed:.1f}s, keeping original", flush=True)
                return text

        # Large file: split by sentences, process, join
        print(f"      [Ollama] Large text ({len(text):,} chars), splitting by sentences...", flush=True)

        chunks = self._create_chunks_by_sentence(text, max_chunk_chars)
        print(f"      [Ollama] Split into {len(chunks)} chunks (max {max_chunk_chars} chars each)", flush=True)

        # Show chunk sizes
        chunk_sizes = [len(c) for c in chunks]
        print(f"      [Ollama] Chunk sizes: {chunk_sizes}", flush=True)

        # Process each chunk
        processed_chunks = []
        total_start = time.time()

        for i, chunk in enumerate(chunks):
            print(f"      [Ollama] Chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)...", flush=True, end=" ")
            start = time.time()
            result = self._call_ollama(chunk)
            elapsed = time.time() - start

            if result:
                print(f"OK ({elapsed:.1f}s)", flush=True)
                processed_chunks.append(result)
            else:
                print(f"FAILED ({elapsed:.1f}s), keeping original", flush=True)
                processed_chunks.append(chunk)

        # Join results with newline (preserve some structure)
        final_result = '\n'.join(processed_chunks)
        total_elapsed = time.time() - total_start
        print(f"      [Ollama] All chunks done in {total_elapsed:.1f}s -> {len(final_result):,} chars", flush=True)

        # Save to cache
        if cache:
            cache.set(text, "ollama", self.model, final_result)

        return final_result

    def is_available(self) -> bool:
        """Check if Ollama is available (server only, doesn't load model)."""
        return self._check_server(force=False)


# =============================================================================
# Main PostProcessor
# =============================================================================

class PostProcessor:
    """
    Main post-processor for Vietnamese text correction.

    Supports different processing methods for direct and indirect extraction.
    """

    def __init__(
        self,
        direct_method: ProcessingMethod = None,
        indirect_method: ProcessingMethod = None
    ):
        """
        Initialize PostProcessor.

        Args:
            direct_method: Method for direct extraction (doc, txt, etc.)
            indirect_method: Method for indirect extraction (OCR, ASR)
        """
        # Read from env with defaults
        self.direct_method = direct_method or os.getenv("POSTPROCESS_DIRECT", "none")
        self.indirect_method = indirect_method or os.getenv("POSTPROCESS_INDIRECT", "ollama")

        # Lazy-loaded processors
        self._transformer = None
        self._ollama = None

        logger.info(
            f"PostProcessor: direct={self.direct_method}, indirect={self.indirect_method}"
        )

    @property
    def transformer(self) -> TransformerProcessor:
        """Lazy load transformer processor."""
        if self._transformer is None:
            self._transformer = TransformerProcessor()
        return self._transformer

    @property
    def ollama(self) -> OllamaProcessor:
        """Lazy load Ollama processor."""
        if self._ollama is None:
            self._ollama = OllamaProcessor()
        return self._ollama

    def process(
        self,
        text: str,
        extraction_type: ExtractionType = "indirect",
        method_override: ProcessingMethod = None,
        sentence_mode: bool = False
    ) -> str:
        """
        Process text based on extraction type.

        Args:
            text: Input text to process
            extraction_type: "direct" or "indirect"
            method_override: Override the default method for this call
            sentence_mode: If True, process each sentence as a separate chunk

        Returns:
            Processed text
        """
        if not text or not text.strip():
            return text

        # Determine method
        if method_override:
            method = method_override
        elif extraction_type == "direct":
            method = self.direct_method
        else:
            method = self.indirect_method

        logger.info(f"Processing: method={method}, type={extraction_type}, len={len(text)}")

        # Apply method
        if method == "none":
            return text
        elif method == "transformer":
            return self.transformer.process(text)
        elif method == "ollama":
            # sentence_mode: each sentence = 1 chunk (max_chunk_chars=1 forces this)
            max_chars = 1 if sentence_mode else 1500
            return self.ollama.process(text, max_chunk_chars=max_chars)
        else:
            logger.warning(f"Unknown method: {method}, returning original text")
            return text

    def get_info(self) -> Dict[str, Any]:
        """Get processor information (doesn't trigger model loading)."""
        return {
            "direct_method": self.direct_method,
            "indirect_method": self.indirect_method,
            "transformer_model": os.getenv(
                "POSTPROCESS_TRANSFORMER_MODEL",
                "bmd1905/vietnamese-correction-v2"
            ),
            "ollama_model": os.getenv("POSTPROCESS_OLLAMA_MODEL", "qwen2.5:7b"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }


# =============================================================================
# Singleton and Convenience Functions
# =============================================================================

_default_processor: Optional[PostProcessor] = None


def get_processor() -> PostProcessor:
    """Get or create default PostProcessor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = PostProcessor()
    return _default_processor


def post_process_text(
    text: str,
    extraction_type: ExtractionType = "indirect",
    method: ProcessingMethod = None
) -> str:
    """
    Quick function to post-process text.

    Args:
        text: Input text
        extraction_type: "direct" (doc, txt) or "indirect" (OCR, ASR)
        method: Override method (none, transformer, ollama)

    Returns:
        Processed text

    Example:
        >>> post_process_text("ĐI HC QUC GIA", extraction_type="indirect")
        "ĐẠI HỌC QUỐC GIA"
    """
    return get_processor().process(text, extraction_type, method)
