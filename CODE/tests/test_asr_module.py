# -*- coding: utf-8 -*-
"""
Test ASR Module - Faster-Whisper & OpenAI-Whisper
=================================================

Comprehensive tests for dual-engine ASR implementation.

Run: python tests/test_asr_module.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestASRModule:
    """Test suite for ASR Module."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def run_test(self, test_func, test_name):
        """Run a single test."""
        try:
            result = test_func()
            if result == "SKIP":
                self.skipped += 1
                self.results.append((test_name, "SKIP", None))
                print(f"  [SKIP] {test_name}")
            else:
                self.passed += 1
                self.results.append((test_name, "PASS", None))
                print(f"  [PASS] {test_name}")
        except Exception as e:
            self.failed += 1
            self.results.append((test_name, "FAIL", str(e)))
            print(f"  [FAIL] {test_name}: {e}")

    # =========================================================================
    # Import Tests
    # =========================================================================

    def test_import_module(self):
        """Test importing ASR module."""
        from src.modules.asr_module import WhisperASR, BaseASR
        assert WhisperASR is not None
        assert BaseASR is not None

    def test_import_faster_whisper_class(self):
        """Test importing FasterWhisperASR class."""
        from src.modules.asr_module import FasterWhisperASR
        assert FasterWhisperASR is not None

    def test_import_openai_whisper_class(self):
        """Test importing OpenAIWhisperASR class."""
        from src.modules.asr_module import OpenAIWhisperASR
        assert OpenAIWhisperASR is not None

    # =========================================================================
    # Factory Function Tests
    # =========================================================================

    def test_factory_faster_engine(self):
        """Test factory function with faster engine."""
        from src.modules.asr_module import WhisperASR, FasterWhisperASR

        try:
            asr = WhisperASR(model_name="tiny", language="vi", engine="faster")
            assert isinstance(asr, FasterWhisperASR)
            assert asr.model_name == "tiny"
            assert asr.language == "vi"
        except ImportError:
            return "SKIP"

    def test_factory_openai_engine(self):
        """Test factory function with openai engine."""
        from src.modules.asr_module import WhisperASR, OpenAIWhisperASR

        try:
            asr = WhisperASR(model_name="tiny", language="vi", engine="openai")
            assert isinstance(asr, OpenAIWhisperASR)
            assert asr.model_name == "tiny"
        except ImportError:
            return "SKIP"

    def test_factory_auto_engine(self):
        """Test factory function with auto engine detection."""
        from src.modules.asr_module import WhisperASR, BaseASR

        asr = WhisperASR(model_name="tiny", language="vi", engine="auto")
        assert isinstance(asr, BaseASR)

    def test_factory_invalid_engine(self):
        """Test factory function with invalid engine raises error."""
        from src.modules.asr_module import WhisperASR

        try:
            WhisperASR(model_name="tiny", engine="invalid_engine")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Unknown engine" in str(e)

    def test_factory_from_env(self):
        """Test factory reads WHISPER_ENGINE from environment."""
        from src.modules.asr_module import WhisperASR

        original = os.environ.get("WHISPER_ENGINE")
        try:
            os.environ["WHISPER_ENGINE"] = "faster"
            asr = WhisperASR(model_name="tiny")
            assert asr.__class__.__name__ == "FasterWhisperASR"
        except ImportError:
            return "SKIP"
        finally:
            if original:
                os.environ["WHISPER_ENGINE"] = original

    # =========================================================================
    # FasterWhisperASR Tests
    # =========================================================================

    def test_faster_whisper_init(self):
        """Test FasterWhisperASR initialization."""
        try:
            from src.modules.asr_module import FasterWhisperASR

            asr = FasterWhisperASR(model_name="tiny", language="vi")
            assert asr.model is not None
            assert asr.model_name == "tiny"
            assert asr.compute_type is not None
        except ImportError:
            return "SKIP"

    def test_faster_whisper_compute_type_cpu(self):
        """Test FasterWhisperASR uses int8 on CPU."""
        try:
            from src.modules.asr_module import FasterWhisperASR

            asr = FasterWhisperASR(model_name="tiny", device="cpu")
            assert asr.compute_type == "int8"
        except ImportError:
            return "SKIP"

    def test_faster_whisper_custom_compute_type(self):
        """Test FasterWhisperASR with custom compute type."""
        try:
            from src.modules.asr_module import FasterWhisperASR

            asr = FasterWhisperASR(model_name="tiny", device="cpu", compute_type="float32")
            assert asr.compute_type == "float32"
        except ImportError:
            return "SKIP"

    # =========================================================================
    # OpenAIWhisperASR Tests
    # =========================================================================

    def test_openai_whisper_init(self):
        """Test OpenAIWhisperASR initialization."""
        try:
            from src.modules.asr_module import OpenAIWhisperASR

            asr = OpenAIWhisperASR(model_name="tiny", language="vi")
            assert asr.model is not None
            assert asr.model_name == "tiny"
        except ImportError:
            return "SKIP"

    # =========================================================================
    # BaseASR Methods Tests
    # =========================================================================

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        from src.modules.asr_module import WhisperASR

        asr = WhisperASR(model_name="tiny", engine="auto")

        # Test various timestamps
        assert asr._format_timestamp(0) == "00:00:00.00"
        assert asr._format_timestamp(61.5) == "00:01:01.50"
        assert asr._format_timestamp(3661.25) == "01:01:01.25"

    def test_format_srt_timestamp(self):
        """Test SRT timestamp formatting."""
        from src.modules.asr_module import WhisperASR

        asr = WhisperASR(model_name="tiny", engine="auto")

        assert asr._format_srt_timestamp(0) == "00:00:00,000"
        assert asr._format_srt_timestamp(61.5) == "00:01:01,500"
        assert asr._format_srt_timestamp(3661.25) == "01:01:01,250"

    def test_save_transcript_json(self):
        """Test saving transcript as JSON."""
        from src.modules.asr_module import WhisperASR
        import json

        asr = WhisperASR(model_name="tiny", engine="auto")

        transcript = {
            "audio_filename": "test.mp3",
            "duration": 10.5,
            "full_text": "Test transcript",
            "segments": [
                {"start": 0, "end": 5, "text": "Hello"},
                {"start": 5, "end": 10, "text": "World"}
            ],
            "transcribed_at": "2024-01-01T00:00:00",
            "engine": "test"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            asr.save_transcript(transcript, output_path, format="json")

            assert output_path.exists()
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["full_text"] == "Test transcript"

    def test_save_transcript_txt(self):
        """Test saving transcript as TXT."""
        from src.modules.asr_module import WhisperASR

        asr = WhisperASR(model_name="tiny", engine="auto")

        transcript = {
            "audio_filename": "test.mp3",
            "duration": 10.5,
            "full_text": "Test transcript",
            "segments": [
                {"start": 0, "end": 5, "text": "Hello"},
                {"start": 5, "end": 10, "text": "World"}
            ],
            "transcribed_at": "2024-01-01T00:00:00",
            "engine": "test"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.txt"
            asr.save_transcript(transcript, output_path, format="txt")

            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "Test transcript" in content
            assert "Hello" in content
            assert "World" in content

    def test_save_transcript_srt(self):
        """Test saving transcript as SRT."""
        from src.modules.asr_module import WhisperASR

        asr = WhisperASR(model_name="tiny", engine="auto")

        transcript = {
            "audio_filename": "test.mp3",
            "duration": 10.5,
            "full_text": "Test transcript",
            "segments": [
                {"start": 0, "end": 5, "text": "Hello"},
                {"start": 5, "end": 10, "text": "World"}
            ],
            "transcribed_at": "2024-01-01T00:00:00",
            "engine": "test"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.srt"
            asr.save_transcript(transcript, output_path, format="srt")

            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "1\n" in content
            assert "Hello" in content
            assert "-->" in content

    # =========================================================================
    # Environment Variable Tests
    # =========================================================================

    def test_env_whisper_model(self):
        """Test WHISPER_MODEL environment variable."""
        from src.modules.asr_module import WhisperASR

        original = os.environ.get("WHISPER_MODEL")
        try:
            os.environ["WHISPER_MODEL"] = "tiny"
            asr = WhisperASR(engine="auto")
            assert asr.model_name == "tiny"
        finally:
            if original:
                os.environ["WHISPER_MODEL"] = original
            else:
                os.environ.pop("WHISPER_MODEL", None)

    def test_env_whisper_device(self):
        """Test WHISPER_DEVICE environment variable."""
        from src.modules.asr_module import WhisperASR

        original = os.environ.get("WHISPER_DEVICE")
        try:
            os.environ["WHISPER_DEVICE"] = "cpu"
            asr = WhisperASR(model_name="tiny", engine="auto")
            assert asr.device == "cpu"
        finally:
            if original:
                os.environ["WHISPER_DEVICE"] = original
            else:
                os.environ.pop("WHISPER_DEVICE", None)

    # =========================================================================
    # Integration with AudioProcessor
    # =========================================================================

    def test_audio_processor_imports_asr(self):
        """Test AudioProcessor can import WhisperASR."""
        from src.modules.document_processor.audio_processor import AudioProcessor

        processor = AudioProcessor()
        # Check that _get_asr method exists
        assert hasattr(processor, "_get_asr")

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all(self):
        """Run all tests."""
        print("\n" + "=" * 60)
        print("ASR MODULE TESTS")
        print("=" * 60)

        tests = [
            # Import Tests
            (self.test_import_module, "Import ASR module"),
            (self.test_import_faster_whisper_class, "Import FasterWhisperASR"),
            (self.test_import_openai_whisper_class, "Import OpenAIWhisperASR"),

            # Factory Function Tests
            (self.test_factory_faster_engine, "Factory: faster engine"),
            (self.test_factory_openai_engine, "Factory: openai engine"),
            (self.test_factory_auto_engine, "Factory: auto engine"),
            (self.test_factory_invalid_engine, "Factory: invalid engine error"),
            (self.test_factory_from_env, "Factory: read from WHISPER_ENGINE env"),

            # FasterWhisperASR Tests
            (self.test_faster_whisper_init, "FasterWhisper: initialization"),
            (self.test_faster_whisper_compute_type_cpu, "FasterWhisper: int8 on CPU"),
            (self.test_faster_whisper_custom_compute_type, "FasterWhisper: custom compute type"),

            # OpenAIWhisperASR Tests
            (self.test_openai_whisper_init, "OpenAIWhisper: initialization"),

            # BaseASR Methods
            (self.test_format_timestamp, "BaseASR: format timestamp"),
            (self.test_format_srt_timestamp, "BaseASR: format SRT timestamp"),
            (self.test_save_transcript_json, "BaseASR: save JSON"),
            (self.test_save_transcript_txt, "BaseASR: save TXT"),
            (self.test_save_transcript_srt, "BaseASR: save SRT"),

            # Environment Variables
            (self.test_env_whisper_model, "Env: WHISPER_MODEL"),
            (self.test_env_whisper_device, "Env: WHISPER_DEVICE"),

            # Integration
            (self.test_audio_processor_imports_asr, "Integration: AudioProcessor imports ASR"),
        ]

        print(f"\nRunning {len(tests)} tests...\n")

        for test_func, test_name in tests:
            self.run_test(test_func, test_name)

        # Summary
        print("\n" + "-" * 60)
        print(f"Results: {self.passed} passed, {self.failed} failed, {self.skipped} skipped")
        print("-" * 60)

        if self.failed > 0:
            print("\nFailed tests:")
            for name, status, error in self.results:
                if status == "FAIL":
                    print(f"  - {name}: {error}")

        return self.failed == 0


def test_transcribe_with_sample_audio():
    """Test actual transcription if sample audio exists."""
    print("\n" + "=" * 60)
    print("TRANSCRIPTION TEST (requires sample audio)")
    print("=" * 60)

    # Check for sample audio files
    sample_paths = [
        "tests/test_data/sample.mp3",
        "tests/test_data/sample.wav",
        "data/audio/sample.mp3",
    ]

    sample_audio = None
    for path in sample_paths:
        if os.path.exists(path):
            sample_audio = path
            break

    if not sample_audio:
        print("  [SKIP] No sample audio file found")
        print("  Create tests/test_data/sample.mp3 to enable this test")
        return True

    print(f"  Using: {sample_audio}")

    try:
        from src.modules.asr_module import WhisperASR

        # Test with faster-whisper
        print("\n  Testing Faster-Whisper transcription...")
        start = time.time()
        asr = WhisperASR(model_name="tiny", engine="faster")
        result = asr.transcribe_audio(sample_audio, verbose=False)
        elapsed = time.time() - start

        print(f"  Duration: {result['duration']:.1f}s")
        print(f"  Segments: {len(result['segments'])}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  RTF: {elapsed/result['duration']:.2f}x")
        print(f"  Text preview: {result['full_text'][:100]}...")
        print("  [PASS] Faster-Whisper transcription")

        return True

    except ImportError as e:
        print(f"  [SKIP] {e}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def compare_engines():
    """Compare speed between faster-whisper and openai-whisper."""
    print("\n" + "=" * 60)
    print("ENGINE COMPARISON (requires sample audio)")
    print("=" * 60)

    sample_audio = None
    for path in ["tests/test_data/sample.mp3", "tests/test_data/sample.wav"]:
        if os.path.exists(path):
            sample_audio = path
            break

    if not sample_audio:
        print("  [SKIP] No sample audio file found")
        return True

    results = {}

    # Test faster-whisper
    try:
        from src.modules.asr_module import FasterWhisperASR

        print("\n  Testing Faster-Whisper...")
        asr = FasterWhisperASR(model_name="tiny")

        start = time.time()
        result = asr.transcribe_audio(sample_audio, verbose=False)
        elapsed = time.time() - start

        results["faster"] = {
            "time": elapsed,
            "duration": result["duration"],
            "rtf": elapsed / result["duration"]
        }
        print(f"  Time: {elapsed:.2f}s, RTF: {results['faster']['rtf']:.2f}x")

    except ImportError:
        print("  [SKIP] faster-whisper not installed")

    # Test openai-whisper
    try:
        from src.modules.asr_module import OpenAIWhisperASR

        print("\n  Testing OpenAI-Whisper...")
        asr = OpenAIWhisperASR(model_name="tiny")

        start = time.time()
        result = asr.transcribe_audio(sample_audio, verbose=False)
        elapsed = time.time() - start

        results["openai"] = {
            "time": elapsed,
            "duration": result["duration"],
            "rtf": elapsed / result["duration"]
        }
        print(f"  Time: {elapsed:.2f}s, RTF: {results['openai']['rtf']:.2f}x")

    except ImportError:
        print("  [SKIP] openai-whisper not installed")

    # Compare
    if "faster" in results and "openai" in results:
        speedup = results["openai"]["time"] / results["faster"]["time"]
        print(f"\n  Speedup: {speedup:.1f}x faster with faster-whisper")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ASR MODULE TEST SUITE")
    print("=" * 60)

    all_passed = True

    # Run main test suite
    test_suite = TestASRModule()
    if not test_suite.run_all():
        all_passed = False

    # Run transcription test
    if not test_transcribe_with_sample_audio():
        all_passed = False

    # Run comparison (optional)
    # compare_engines()

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
