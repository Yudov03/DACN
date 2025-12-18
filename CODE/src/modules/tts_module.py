"""
Text-to-Speech Module
=====================

Su dung edge-tts de chuyen text thanh audio.
Ho tro tieng Viet va nhieu ngon ngu khac.

Usage:
    from src.modules.tts_module import TextToSpeech, text_to_speech

    # Basic usage
    tts = TextToSpeech(voice="vi-female")
    audio_bytes = tts.synthesize_sync("Xin chao!")

    # Save to file
    path = tts.synthesize_to_file_sync("Hello", "output.mp3")

    # Quick function
    path = text_to_speech("Xin chao!", voice="vi-female")
"""

import asyncio
import tempfile
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("Warning: edge-tts not installed. Run: pip install edge-tts")

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VoiceInfo:
    """Information about a TTS voice"""
    id: str
    name: str
    language: str
    gender: str
    locale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "locale": self.locale
        }


@dataclass
class TTSResult:
    """Result of TTS synthesis"""
    audio_data: bytes
    duration_ms: Optional[int] = None
    voice_used: str = ""
    text_length: int = 0

    @property
    def success(self) -> bool:
        return len(self.audio_data) > 0


# =============================================================================
# TextToSpeech Class
# =============================================================================

class TextToSpeech:
    """
    Text-to-Speech module using edge-tts.
    Supports Vietnamese and many other languages.

    Features:
    - Multiple voices (Vietnamese, English, etc.)
    - Adjustable rate, volume, pitch
    - Sync and async APIs
    - Streaming support
    """

    # Voice shortcuts
    VOICES = {
        # Vietnamese
        "vi-female": "vi-VN-HoaiMyNeural",
        "vi-male": "vi-VN-NamMinhNeural",
        # English US
        "en-female": "en-US-JennyNeural",
        "en-male": "en-US-GuyNeural",
        # English UK
        "en-gb-female": "en-GB-SoniaNeural",
        "en-gb-male": "en-GB-RyanNeural",
        # Chinese
        "zh-female": "zh-CN-XiaoxiaoNeural",
        "zh-male": "zh-CN-YunxiNeural",
        # Japanese
        "ja-female": "ja-JP-NanamiNeural",
        "ja-male": "ja-JP-KeitaNeural",
        # Korean
        "ko-female": "ko-KR-SunHiNeural",
        "ko-male": "ko-KR-InJoonNeural",
    }

    # Voice display names for UI
    VOICE_NAMES = {
        "vi-female": "Hoai My (Vietnamese Female)",
        "vi-male": "Nam Minh (Vietnamese Male)",
        "en-female": "Jenny (US English Female)",
        "en-male": "Guy (US English Male)",
        "en-gb-female": "Sonia (British Female)",
        "en-gb-male": "Ryan (British Male)",
        "zh-female": "Xiaoxiao (Chinese Female)",
        "zh-male": "Yunxi (Chinese Male)",
        "ja-female": "Nanami (Japanese Female)",
        "ja-male": "Keita (Japanese Male)",
        "ko-female": "SunHi (Korean Female)",
        "ko-male": "InJoon (Korean Male)",
    }

    def __init__(
        self,
        voice: str = "vi-female",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ):
        """
        Initialize TTS engine.

        Args:
            voice: Voice ID or shortcut (vi-female, vi-male, en-female, en-male, etc.)
            rate: Speaking rate (e.g., "+10%", "-20%", "slow", "fast")
            volume: Volume level (e.g., "+50%", "-10%")
            pitch: Voice pitch (e.g., "+10Hz", "-5Hz")
        """
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts is required. Install with: pip install edge-tts")

        self.voice = self._resolve_voice(voice)
        self.rate = self._normalize_rate(rate)
        self.volume = volume
        self.pitch = pitch

    def _resolve_voice(self, voice: str) -> str:
        """Resolve voice shortcut to full voice ID"""
        return self.VOICES.get(voice, voice)

    def _normalize_rate(self, rate: str) -> str:
        """Normalize rate value"""
        rate_map = {
            "slow": "-20%",
            "normal": "+0%",
            "fast": "+20%",
            "very-slow": "-40%",
            "very-fast": "+40%"
        }
        return rate_map.get(rate.lower(), rate)

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to audio bytes (async).

        Args:
            text: Text to convert

        Returns:
            Audio data as bytes (MP3 format)
        """
        if not text.strip():
            return b""

        try:
            communicate = edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            return audio_data

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise TTSError(f"Failed to synthesize text: {e}")

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str
    ) -> str:
        """
        Convert text to audio file (async).

        Args:
            text: Text to convert
            output_path: Output file path (MP3)

        Returns:
            Path to output file
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            communicate = edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )

            await communicate.save(output_path)
            return output_path

        except Exception as e:
            logger.error(f"TTS save failed: {e}")
            raise TTSError(f"Failed to save audio: {e}")

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks for real-time playback (async).

        Args:
            text: Text to convert

        Yields:
            Audio chunks as bytes
        """
        if not text.strip():
            return

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize_with_timing(self, text: str) -> TTSResult:
        """
        Synthesize with timing information.

        Args:
            text: Text to convert

        Returns:
            TTSResult with audio data and metadata
        """
        import time
        start = time.time()

        audio_data = await self.synthesize(text)

        return TTSResult(
            audio_data=audio_data,
            duration_ms=int((time.time() - start) * 1000),
            voice_used=self.voice,
            text_length=len(text)
        )

    # =========================================================================
    # Sync Methods (for convenience)
    # =========================================================================

    def synthesize_sync(self, text: str) -> bytes:
        """
        Synchronous version of synthesize.

        Args:
            text: Text to convert

        Returns:
            Audio data as bytes
        """
        return self._run_async(self.synthesize(text))

    def synthesize_to_file_sync(self, text: str, output_path: str) -> str:
        """
        Synchronous version of synthesize_to_file.

        Args:
            text: Text to convert
            output_path: Output file path

        Returns:
            Path to output file
        """
        return self._run_async(self.synthesize_to_file(text, output_path))

    def _run_async(self, coro):
        """Run async coroutine synchronously"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new loop
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(coro)

    # =========================================================================
    # Voice Management
    # =========================================================================

    @classmethod
    async def list_voices(cls, language: str = None) -> List[VoiceInfo]:
        """
        List available voices (async).

        Args:
            language: Filter by language code (e.g., "vi", "en")

        Returns:
            List of VoiceInfo objects
        """
        if not EDGE_TTS_AVAILABLE:
            return []

        try:
            voices = await edge_tts.list_voices()

            result = []
            for v in voices:
                locale = v.get("Locale", "")

                # Filter by language if specified
                if language:
                    if not locale.lower().startswith(language.lower()):
                        continue

                result.append(VoiceInfo(
                    id=v.get("ShortName", ""),
                    name=v.get("FriendlyName", ""),
                    language=v.get("Locale", ""),
                    gender=v.get("Gender", ""),
                    locale=locale
                ))

            return result

        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []

    @classmethod
    def list_voices_sync(cls, language: str = None) -> List[VoiceInfo]:
        """Synchronous version of list_voices."""
        try:
            return asyncio.run(cls.list_voices(language))
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []

    @classmethod
    def get_available_shortcuts(cls) -> Dict[str, str]:
        """Get available voice shortcuts"""
        return cls.VOICES.copy()

    @classmethod
    def get_voice_display_names(cls) -> Dict[str, str]:
        """Get voice display names for UI"""
        return cls.VOICE_NAMES.copy()

    # =========================================================================
    # Settings
    # =========================================================================

    def set_voice(self, voice: str):
        """
        Change voice.

        Args:
            voice: Voice ID or shortcut
        """
        self.voice = self._resolve_voice(voice)

    def set_rate(self, rate: str):
        """
        Change speaking rate.

        Args:
            rate: Rate value (e.g., "+10%", "-20%", "slow", "fast")
        """
        self.rate = self._normalize_rate(rate)

    def set_volume(self, volume: str):
        """
        Change volume.

        Args:
            volume: Volume value (e.g., "+50%", "-10%")
        """
        self.volume = volume

    def set_pitch(self, pitch: str):
        """
        Change pitch.

        Args:
            pitch: Pitch value (e.g., "+10Hz", "-5Hz")
        """
        self.pitch = pitch

    def get_settings(self) -> Dict[str, str]:
        """Get current settings"""
        return {
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume,
            "pitch": self.pitch
        }


# =============================================================================
# Exceptions
# =============================================================================

class TTSError(Exception):
    """TTS-related error"""
    pass


# =============================================================================
# Convenience Functions
# =============================================================================

def text_to_speech(
    text: str,
    voice: str = "vi-female",
    output_path: str = None,
    rate: str = "+0%"
) -> str:
    """
    Quick function to convert text to speech file.

    Args:
        text: Text to convert
        voice: Voice shortcut (vi-female, vi-male, en-female, en-male)
        output_path: Output file path (optional, creates temp file if None)
        rate: Speaking rate

    Returns:
        Path to audio file
    """
    tts = TextToSpeech(voice=voice, rate=rate)

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp3")

    return tts.synthesize_to_file_sync(text, output_path)


def text_to_audio_bytes(
    text: str,
    voice: str = "vi-female",
    rate: str = "+0%"
) -> bytes:
    """
    Quick function to get audio bytes.

    Args:
        text: Text to convert
        voice: Voice shortcut
        rate: Speaking rate

    Returns:
        Audio data as bytes (MP3 format)
    """
    tts = TextToSpeech(voice=voice, rate=rate)
    return tts.synthesize_sync(text)


def get_available_voices(language: str = None) -> List[VoiceInfo]:
    """
    Get list of available voices.

    Args:
        language: Filter by language (e.g., "vi", "en")

    Returns:
        List of VoiceInfo
    """
    return TextToSpeech.list_voices_sync(language)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    import sys

    # Fix Windows encoding
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("   TTS MODULE TEST")
    print("=" * 60)

    # Test 1: List voices
    print("\n[1] Available Vietnamese voices:")
    voices = get_available_voices("vi")
    for v in voices:
        print(f"  - {v.id}: {v.name} ({v.gender})")

    # Test 2: Synthesize
    print("\n[2] Testing synthesis...")
    try:
        audio = text_to_audio_bytes("Xin chao, toi la tro ly AI.", voice="vi-female")
        print(f"  Generated {len(audio)} bytes of audio")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 3: Save to file
    print("\n[3] Testing save to file...")
    try:
        path = text_to_speech("Hello, I am an AI assistant.", voice="en-female")
        print(f"  Saved to: {path}")
        # Cleanup
        Path(path).unlink()
        print("  Cleaned up temp file")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("   TEST COMPLETE")
    print("=" * 60)
