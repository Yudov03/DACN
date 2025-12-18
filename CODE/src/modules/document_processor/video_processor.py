"""
Video Processor - Extract text from video files using Whisper ASR.
Extracts audio track and transcribes it.
Supports: .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm
"""

import time
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import os

from .base import (
    BaseProcessor,
    ProcessedDocument,
    DocumentMetadata,
    TextChunk,
)


class VideoProcessor(BaseProcessor):
    """
    Processor cho Video files.

    Features:
    - Extract audio từ video using FFmpeg
    - Transcribe using Whisper ASR
    - Support multiple video formats
    - Timestamps cho từng segment
    """

    VERSION = "1.0"

    def __init__(self, config: Dict = None):
        """
        Initialize Video Processor.

        Args:
            config: Configuration dict with options:
                - whisper_model: Whisper model name (default: 'base')
                - language: Language code (default: 'vi')
                - device: 'cuda' or 'cpu' (default: auto-detect)
                - return_timestamps: Include timestamps (default: True)
                - keep_audio: Keep extracted audio file (default: False)
        """
        super().__init__(config)
        self.whisper_model = self.config.get("whisper_model", "base")
        self.language = self.config.get("language", "vi")
        self.device = self.config.get("device", None)
        self.return_timestamps = self.config.get("return_timestamps", True)
        self.keep_audio = self.config.get("keep_audio", False)
        self._asr = None

    def supported_extensions(self) -> List[str]:
        return [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"]

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
                    "Whisper not installed. Run: pip install openai-whisper"
                )
        return self._asr

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available and add to PATH if needed"""
        # Check if already available
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
                    bin_path = os.path.join(winget_base, pkg_dir, "ffmpeg-8.0.1-full_build", "bin")
                    if os.path.exists(bin_path):
                        common_paths.insert(0, bin_path)

        for path in common_paths:
            if os.path.isdir(path):
                ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
                if os.path.exists(ffmpeg_exe):
                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                    return True

        return False

    def _extract_audio_ffmpeg(
        self, video_path: str, output_path: str
    ) -> bool:
        """
        Extract audio from video using FFmpeg.

        Args:
            video_path: Path to video file
            output_path: Path for extracted audio

        Returns:
            True if successful
        """
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # WAV format
                "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                output_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,  # 5 minute timeout
                errors='replace'  # Handle encoding errors on Windows
            )

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print("FFmpeg timeout - video may be too long")
            return False
        except Exception as e:
            print(f"FFmpeg error: {e}")
            return False

    def _extract_audio_moviepy(
        self, video_path: str, output_path: str
    ) -> bool:
        """
        Extract audio using moviepy (fallback if FFmpeg not available).

        Args:
            video_path: Path to video file
            output_path: Path for extracted audio

        Returns:
            True if successful
        """
        try:
            from moviepy.editor import VideoFileClip

            video = VideoFileClip(video_path)
            audio = video.audio

            if audio is None:
                print("Video has no audio track")
                return False

            audio.write_audiofile(
                output_path,
                fps=16000,
                nbytes=2,
                codec='pcm_s16le',
                verbose=False,
                logger=None
            )

            video.close()
            return True

        except ImportError:
            raise ImportError(
                "Neither FFmpeg nor moviepy available. "
                "Install FFmpeg or run: pip install moviepy"
            )
        except Exception as e:
            print(f"Moviepy error: {e}")
            return False

    def _get_video_info(self, video_path: str) -> Dict:
        """Get video metadata using FFprobe or moviepy"""
        info = {
            "duration": 0,
            "width": 0,
            "height": 0,
            "fps": 0,
            "has_audio": True,
        }

        # Try FFprobe first (use same path-finding logic as _check_ffmpeg)
        ffprobe_available = shutil.which("ffprobe")
        if not ffprobe_available and self._check_ffmpeg():
            # ffmpeg was found, ffprobe should be in same directory
            ffprobe_available = shutil.which("ffprobe")

        if ffprobe_available:
            try:
                cmd = [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    video_path
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)

                    # Get format info
                    if "format" in data:
                        info["duration"] = float(
                            data["format"].get("duration", 0)
                        )

                    # Get stream info
                    has_audio_stream = False
                    for stream in data.get("streams", []):
                        if stream.get("codec_type") == "video":
                            info["width"] = stream.get("width", 0)
                            info["height"] = stream.get("height", 0)
                            fps_str = stream.get("r_frame_rate", "0/1")
                            if "/" in fps_str:
                                num, den = fps_str.split("/")
                                info["fps"] = float(num) / float(den) if float(den) > 0 else 0
                        elif stream.get("codec_type") == "audio":
                            has_audio_stream = True

                    info["has_audio"] = has_audio_stream
                    return info
            except Exception as e:
                print(f"FFprobe warning: {e}")

        # Fallback to moviepy
        try:
            from moviepy.editor import VideoFileClip
            video = VideoFileClip(video_path)
            info["duration"] = video.duration or 0
            info["width"] = video.size[0] if video.size else 0
            info["height"] = video.size[1] if video.size else 0
            info["fps"] = video.fps or 0
            info["has_audio"] = video.audio is not None
            video.close()
        except Exception:
            pass

        return info

    def process(self, file_path: str) -> ProcessedDocument:
        """Process video file: extract audio and transcribe"""
        start_time = time.time()

        try:
            self.validate(file_path)
        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

        # Get video info
        video_info = self._get_video_info(file_path)

        if not video_info.get("has_audio", True):
            return self._create_error_result(
                file_path,
                ValueError("Video has no audio track"),
                start_time
            )

        # Create temp file for extracted audio
        temp_audio = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as f:
                temp_audio = f.name

            # Extract audio (prefer FFmpeg)
            print(f"Extracting audio from video: {Path(file_path).name}...")

            if self._check_ffmpeg():
                success = self._extract_audio_ffmpeg(file_path, temp_audio)
            else:
                success = self._extract_audio_moviepy(file_path, temp_audio)

            if not success:
                return self._create_error_result(
                    file_path,
                    RuntimeError("Failed to extract audio from video"),
                    start_time
                )

            # Transcribe using Whisper
            print("Transcribing audio...")
            asr = self._get_asr()
            result = asr.transcribe_audio(
                temp_audio,
                return_timestamps=self.return_timestamps,
                verbose=False
            )

            # Extract full text
            full_text = result.get("full_text", "")

            # Create chunks from segments
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
            metadata = self._extract_video_metadata(
                file_path, video_info, result
            )

            return ProcessedDocument(
                content=full_text,
                chunks=chunks,
                metadata=metadata,
                source_file=file_path,
                file_type="video",
                processed_at=datetime.now(),
                processing_time=time.time() - start_time,
                processor_version=self.VERSION,
                success=True,
                extraction_type="indirect"  # ASR always produces indirect extraction
            )

        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

        finally:
            # Cleanup temp audio file
            if temp_audio and not self.keep_audio:
                try:
                    Path(temp_audio).unlink(missing_ok=True)
                except Exception:
                    pass

    def _extract_video_metadata(
        self,
        file_path: str,
        video_info: Dict,
        asr_result: Dict
    ) -> DocumentMetadata:
        """Extract video-specific metadata"""
        base_meta = self.extract_metadata(file_path)

        duration = video_info.get("duration", 0)
        segments = asr_result.get("segments", [])

        base_meta.extra = {
            "duration_seconds": duration,
            "duration_formatted": self._format_duration(duration),
            "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
            "fps": video_info.get("fps", 0),
            "segment_count": len(segments),
            "language_detected": asr_result.get("language", self.language),
            "whisper_model": self.whisper_model,
            "media_type": "video",
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
