# -*- coding: utf-8 -*-
"""
ASR Module - Automatic Speech Recognition
=========================================

Supports two engines:
- faster-whisper (default): 4x faster, same accuracy, CTranslate2 backend
- openai-whisper: Original OpenAI implementation, PyTorch backend

Configuration (.env):
    WHISPER_ENGINE=faster       # faster, openai (default: faster)
    WHISPER_MODEL=base          # tiny, base, small, medium, large-v2, large-v3
    WHISPER_DEVICE=cuda         # cuda, cpu, auto
    WHISPER_COMPUTE_TYPE=auto   # float16, int8, int8_float16, auto (faster-whisper only)
"""

import os
import sys
import warnings
import threading

# Fix Windows encoding for subprocess (FFmpeg output)
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    _original_excepthook = threading.excepthook

    def _silent_excepthook(args):
        if args.exc_type == UnicodeDecodeError:
            return
        _original_excepthook(args)

    threading.excepthook = _silent_excepthook

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod


# =============================================================================
# Base ASR Class
# =============================================================================

class BaseASR(ABC):
    """Abstract base class for ASR engines."""

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        language: str = "vi"
    ):
        import torch

        self.model_name = model_name or os.getenv("WHISPER_MODEL", "base")
        self.language = language

        # Device detection
        if device is None:
            device = os.getenv("WHISPER_DEVICE")
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None

    @abstractmethod
    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        return_timestamps: bool = True,
        verbose: bool = True
    ) -> Dict:
        """Transcribe audio to text with timestamps."""
        pass

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS.mm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def save_transcript(
        self,
        transcript_data: Dict,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> Path:
        """Save transcript to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Audio: {transcript_data['audio_filename']}\n")
                f.write(f"Duration: {transcript_data['duration']:.2f}s\n")
                f.write(f"Engine: {transcript_data.get('engine', 'whisper')}\n")
                f.write(f"Transcribed at: {transcript_data['transcribed_at']}\n")
                f.write("=" * 80 + "\n\n")
                f.write(transcript_data['full_text'] + "\n\n")
                f.write("=" * 80 + "\n")
                f.write("SEGMENTS WITH TIMESTAMPS:\n")
                f.write("=" * 80 + "\n\n")

                for segment in transcript_data['segments']:
                    start_time = self._format_timestamp(segment['start'])
                    end_time = self._format_timestamp(segment['end'])
                    f.write(f"[{start_time} --> {end_time}]\n")
                    f.write(f"{segment['text']}\n\n")

        elif format == "srt":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(transcript_data['segments'], 1):
                    start = self._format_srt_timestamp(segment['start'])
                    end = self._format_srt_timestamp(segment['end'])
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{segment['text']}\n\n")

        print(f"Saved transcript: {output_path}")
        return output_path

    def transcribe_batch(
        self,
        audio_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_format: str = "json"
    ) -> List[Dict]:
        """Transcribe multiple audio files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_transcripts = []

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {Path(audio_file).name}")

            try:
                transcript_data = self.transcribe_audio(audio_file)
                all_transcripts.append(transcript_data)

                output_filename = Path(audio_file).stem + f"_transcript.{save_format}"
                output_path = output_dir / output_filename
                self.save_transcript(transcript_data, output_path, format=save_format)

            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

        print(f"\nDone! Transcribed {len(all_transcripts)}/{len(audio_files)} files")
        return all_transcripts


# =============================================================================
# Faster-Whisper Engine (CTranslate2)
# =============================================================================

class FasterWhisperASR(BaseASR):
    """
    ASR using Faster-Whisper (CTranslate2 backend).

    4x faster than OpenAI Whisper with same accuracy.
    Supports int8 quantization for CPU, float16 for GPU.
    """

    COMPUTE_TYPES = {
        "cuda": "float16",
        "cpu": "int8",
    }

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        language: str = "vi",
        compute_type: str = None
    ):
        super().__init__(model_name, device, language)

        # Compute type
        if compute_type is None:
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE")
        if compute_type is None or compute_type == "auto":
            compute_type = self.COMPUTE_TYPES.get(self.device, "auto")

        self.compute_type = compute_type

        print(f"Loading Faster-Whisper '{self.model_name}' on {self.device} ({compute_type})...")

        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=compute_type
        )

        print(f"Loaded Faster-Whisper '{self.model_name}'")

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        return_timestamps: bool = True,
        verbose: bool = True,
        beam_size: int = 5,
        vad_filter: bool = None
    ) -> Dict:
        """
        Transcribe audio with Faster-Whisper.

        Args:
            audio_path: Path to audio file
            return_timestamps: Include word timestamps
            verbose: Show progress
            beam_size: Beam size for decoding (default: 5)
            vad_filter: Enable VAD filtering (default: from env)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if verbose:
            print(f"Transcribing: {audio_path.name}...")

        # VAD filter from env if not specified
        if vad_filter is None:
            vad_filter = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"

        # VAD parameters from env
        vad_params = None
        if vad_filter:
            vad_params = dict(
                threshold=float(os.getenv("WHISPER_VAD_THRESHOLD", "0.5")),
                min_speech_duration_ms=int(os.getenv("WHISPER_VAD_MIN_SPEECH_MS", "250")),
                min_silence_duration_ms=int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "500")),
                speech_pad_ms=int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "400"))
            )
            if verbose:
                print(f"VAD enabled: threshold={vad_params['threshold']}, "
                      f"min_speech={vad_params['min_speech_duration_ms']}ms, "
                      f"min_silence={vad_params['min_silence_duration_ms']}ms")

        # Transcribe
        segments_generator, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=beam_size,
            word_timestamps=return_timestamps,
            vad_filter=vad_filter,
            vad_parameters=vad_params
        )

        # Process segments
        segments = []
        full_text_parts = []

        for segment in segments_generator:
            seg_data = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "duration": segment.end - segment.start,
            }

            # Word-level timestamps
            if return_timestamps and segment.words:
                seg_data["words"] = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability
                    }
                    for w in segment.words
                ]

            segments.append(seg_data)
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)
        duration = segments[-1]["end"] if segments else 0

        transcript_data = {
            "audio_file": str(audio_path),
            "audio_filename": audio_path.name,
            "model": self.model_name,
            "language": info.language,
            "language_probability": info.language_probability,
            "full_text": full_text,
            "segments": segments,
            "transcribed_at": datetime.now().isoformat(),
            "duration": duration,
            "engine": "faster-whisper",
            "compute_type": self.compute_type,
            "vad_filter": vad_filter
        }

        if verbose:
            print(f"Done! {len(segments)} segments, {duration:.1f}s")

        return transcript_data


# =============================================================================
# OpenAI Whisper Engine (PyTorch)
# =============================================================================

class OpenAIWhisperASR(BaseASR):
    """
    ASR using OpenAI Whisper (PyTorch backend).

    Original implementation, good for compatibility.
    """

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        language: str = "vi"
    ):
        super().__init__(model_name, device, language)

        print(f"Loading OpenAI Whisper '{self.model_name}' on {self.device}...")

        import whisper

        self.model = whisper.load_model(self.model_name, device=self.device)

        print(f"Loaded OpenAI Whisper '{self.model_name}'")

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        return_timestamps: bool = True,
        verbose: bool = True
    ) -> Dict:
        """Transcribe audio with OpenAI Whisper."""
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if verbose:
            print(f"Transcribing: {audio_path.name}...")

        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            verbose=verbose,
            word_timestamps=return_timestamps
        )

        # Process segments
        segments = []
        for idx, segment in enumerate(result.get("segments", [])):
            segments.append({
                "id": idx,
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip(),
                "duration": segment.get("end", 0.0) - segment.get("start", 0.0)
            })

        duration = segments[-1]["end"] if segments else 0

        transcript_data = {
            "audio_file": str(audio_path),
            "audio_filename": audio_path.name,
            "model": self.model_name,
            "language": result.get("language", self.language),
            "full_text": result["text"],
            "segments": segments,
            "transcribed_at": datetime.now().isoformat(),
            "duration": duration,
            "engine": "openai-whisper"
        }

        if verbose:
            print(f"Done! {len(segments)} segments, {duration:.1f}s")

        return transcript_data


# =============================================================================
# Factory Function
# =============================================================================

def WhisperASR(
    model_name: str = None,
    device: Optional[str] = None,
    language: str = "vi",
    engine: str = None,
    **kwargs
) -> BaseASR:
    """
    Create a Whisper ASR instance.

    Factory function that returns the appropriate engine based on config.

    Args:
        model_name: Model name (tiny, base, small, medium, large-v2, large-v3)
        device: Device (cuda, cpu, auto)
        language: Language code (vi, en, etc.)
        engine: Engine to use (faster, openai, auto)
        **kwargs: Additional arguments for specific engine

    Returns:
        BaseASR instance (FasterWhisperASR or OpenAIWhisperASR)

    Environment:
        WHISPER_ENGINE: faster, openai (default: faster)
    """
    if engine is None:
        engine = os.getenv("WHISPER_ENGINE", "faster").lower()

    # Auto-detect: try faster-whisper first, fallback to openai
    if engine == "auto":
        try:
            import faster_whisper
            engine = "faster"
        except ImportError:
            try:
                import whisper
                engine = "openai"
            except ImportError:
                raise ImportError(
                    "No Whisper engine found. Install either:\n"
                    "  pip install faster-whisper  (recommended)\n"
                    "  pip install openai-whisper"
                )

    if engine == "faster":
        try:
            return FasterWhisperASR(
                model_name=model_name,
                device=device,
                language=language,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

    elif engine == "openai":
        try:
            return OpenAIWhisperASR(
                model_name=model_name,
                device=device,
                language=language
            )
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )

    else:
        raise ValueError(
            f"Unknown engine: {engine}. Use 'faster', 'openai', or 'auto'"
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing ASR Module...")
    print(f"Engine from env: {os.getenv('WHISPER_ENGINE', 'faster')}")

    # Test factory function
    try:
        asr = WhisperASR(model_name="tiny", language="vi")
        print(f"Created: {type(asr).__name__}")
        print("ASR Module initialized successfully!")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")
