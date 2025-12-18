"""
Audio Processor - Extract text from audio files using Whisper ASR.
Supports: .mp3, .wav, .m4a, .flac, .ogg, .wma
"""

import time
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    BaseProcessor,
    ProcessedDocument,
    DocumentMetadata,
    TextChunk,
)


class AudioProcessor(BaseProcessor):
    """
    Processor cho Audio files sử dụng Whisper ASR.

    Features:
    - Transcribe audio to text với timestamps
    - Support multiple audio formats
    - Vietnamese và English language support
    - Tự động tạo chunks theo segments
    """

    VERSION = "1.0"

    def __init__(self, config: Dict = None):
        """
        Initialize Audio Processor.

        Args:
            config: Configuration dict with options:
                - whisper_model: Whisper model name (default: 'base')
                - language: Language code (default: 'vi')
                - device: 'cuda' or 'cpu' (default: auto-detect)
                - return_timestamps: Include timestamps (default: True)
        """
        super().__init__(config)
        self.whisper_model = self.config.get("whisper_model", "base")
        self.language = self.config.get("language", "vi")
        self.device = self.config.get("device", None)
        self.return_timestamps = self.config.get("return_timestamps", True)
        self._asr = None

    def supported_extensions(self) -> List[str]:
        return [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"]

    def _setup_ffmpeg(self) -> bool:
        """Setup FFmpeg PATH if needed (required by Whisper)"""
        if shutil.which("ffmpeg"):
            return True

        # Check FFMPEG_PATH environment variable
        ffmpeg_path = os.environ.get("FFMPEG_PATH")
        if ffmpeg_path and os.path.isdir(ffmpeg_path):
            os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg"):
                return True

        # Check common Windows installation paths
        common_paths = [
            r"C:\Users\{}\AppData\Local\Microsoft\WinGet\Links".format(os.environ.get("USERNAME", "")),
            r"C:\ffmpeg\bin",
            r"C:\Program Files\FFmpeg\bin",
        ]

        # Also check WinGet packages folder
        winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
        if os.path.exists(winget_base):
            for pkg_dir in os.listdir(winget_base):
                if "ffmpeg" in pkg_dir.lower():
                    # Check for various ffmpeg versions
                    for subdir in os.listdir(os.path.join(winget_base, pkg_dir)):
                        bin_path = os.path.join(winget_base, pkg_dir, subdir, "bin")
                        if os.path.exists(bin_path):
                            common_paths.insert(0, bin_path)

        for path in common_paths:
            if os.path.isdir(path):
                ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
                if os.path.exists(ffmpeg_exe):
                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                    return True

        return False

    def _get_asr(self):
        """Lazy load Whisper ASR"""
        if self._asr is None:
            try:
                from ..asr_module import WhisperASR
                self._asr = WhisperASR(
                    model_name=self.whisper_model,
                    device=self.device,
                    language=self.language
                )
            except ImportError:
                raise ImportError(
                    "Whisper not installed. Run:\n"
                    "  pip install faster-whisper  (recommended, 4x faster)\n"
                    "  pip install openai-whisper  (original)"
                )
        return self._asr

    def process(self, file_path: str) -> ProcessedDocument:
        """Process audio file using Whisper ASR"""
        start_time = time.time()

        try:
            self.validate(file_path)
        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

        # Setup FFmpeg (required by Whisper for audio decoding)
        if not self._setup_ffmpeg():
            return self._create_error_result(
                file_path,
                RuntimeError(
                    "FFmpeg not found. Install with: winget install ffmpeg (Windows) "
                    "or brew install ffmpeg (Mac)"
                ),
                start_time
            )

        try:
            # Get ASR and transcribe
            asr = self._get_asr()
            result = asr.transcribe_audio(
                file_path,
                return_timestamps=self.return_timestamps,
                verbose=False
            )

            # Extract full text
            full_text = result.get("full_text", "")

            # Create chunks from segments (with timestamps)
            chunks = []
            segments = result.get("segments", [])
            char_offset = 0

            for i, segment in enumerate(segments):
                segment_text = segment.get("text", "").strip()
                if segment_text:
                    chunk = TextChunk(
                        text=segment_text,
                        start_char=char_offset,
                        end_char=char_offset + len(segment_text),
                        chunk_index=i,
                        metadata={
                            "start_time": segment.get("start", 0),
                            "end_time": segment.get("end", 0),
                            "confidence": segment.get("confidence", 1.0),
                        }
                    )
                    chunks.append(chunk)
                    char_offset += len(segment_text) + 1

            # Build metadata
            metadata = self._extract_audio_metadata(file_path, result)

            return ProcessedDocument(
                content=full_text,
                chunks=chunks,
                metadata=metadata,
                source_file=file_path,
                file_type="audio",
                processed_at=datetime.now(),
                processing_time=time.time() - start_time,
                processor_version=self.VERSION,
                success=True,
                extraction_type="indirect"  # ASR always produces indirect extraction
            )

        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

    def _extract_audio_metadata(
        self, file_path: str, asr_result: Dict
    ) -> DocumentMetadata:
        """Extract audio-specific metadata"""
        base_meta = self.extract_metadata(file_path)

        # Add audio-specific info
        duration = asr_result.get("duration", 0)
        segments = asr_result.get("segments", [])

        base_meta.extra = {
            "duration_seconds": duration,
            "duration_formatted": self._format_duration(duration),
            "segment_count": len(segments),
            "language_detected": asr_result.get("language", self.language),
            "whisper_model": self.whisper_model,
            "media_type": "audio",
        }

        return base_meta

    def _format_duration(self, seconds: float) -> str:
        """Format duration to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def format_transcript_with_timestamps(chunks: List[TextChunk]) -> str:
    """
    Format transcript with timestamps for display.

    Args:
        chunks: List of TextChunk with timestamp metadata

    Returns:
        Formatted string with timestamps
    """
    lines = []
    for chunk in chunks:
        start = chunk.metadata.get("start_time", 0)
        end = chunk.metadata.get("end_time", 0)

        # Format timestamp
        start_str = f"{int(start // 60):02d}:{int(start % 60):02d}"
        end_str = f"{int(end // 60):02d}:{int(end % 60):02d}"

        lines.append(f"[{start_str} - {end_str}] {chunk.text}")

    return "\n".join(lines)
