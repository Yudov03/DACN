"""
Demo script for Text-to-Speech Module
======================================

Demonstrates:
- Voice options (Vietnamese, English)
- Speech synthesis
- Rate and volume control
- Saving to file

Usage:
    python scripts/demo_tts.py
    python scripts/demo_tts.py --text "Xin chào"
    python scripts/demo_tts.py --save output.mp3
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demo_voice_options():
    """Demo available voice options."""
    print_section("1. AVAILABLE VOICES")

    from modules import TextToSpeech

    # Voice shortcuts
    print("\n--- Voice Shortcuts ---")
    shortcuts = TextToSpeech.get_available_shortcuts()
    for shortcut, full_name in shortcuts.items():
        print(f"  {shortcut:12} -> {full_name}")

    # List all voices (async)
    print("\n--- All Available Voices (sample) ---")
    try:
        voices = TextToSpeech.list_voices_sync()
        vi_voices = [v for v in voices if "vi-VN" in v.id]
        en_voices = [v for v in voices if "en-US" in v.id][:5]

        print("\nVietnamese voices:")
        for v in vi_voices:
            print(f"  - {v.id}: {v.gender}")

        print("\nEnglish voices (top 5):")
        for v in en_voices:
            print(f"  - {v.id}: {v.gender}")

    except Exception as e:
        print(f"Error listing voices: {e}")


def demo_synthesis():
    """Demo speech synthesis."""
    print_section("2. SPEECH SYNTHESIS")

    from modules import TextToSpeech

    test_cases = [
        {
            "voice": "vi-female",
            "text": "Xin chào! Đây là hệ thống truy xuất thông tin.",
            "lang": "Vietnamese"
        },
        {
            "voice": "vi-male",
            "text": "Học phí năm 2024 là 15 triệu đồng mỗi kỳ.",
            "lang": "Vietnamese"
        },
        {
            "voice": "en-female",
            "text": "Hello! This is the information retrieval system.",
            "lang": "English"
        },
        {
            "voice": "en-male",
            "text": "Machine learning enables computers to learn from data.",
            "lang": "English"
        }
    ]

    for case in test_cases:
        print(f"\n--- {case['lang']} ({case['voice']}) ---")
        print(f"Text: {case['text']}")

        try:
            tts = TextToSpeech(voice=case['voice'])
            audio = tts.synthesize_sync(case['text'])

            print(f"Voice: {tts.voice}")
            print(f"Audio size: {len(audio):,} bytes")
            print(f"Status: SUCCESS")

        except Exception as e:
            print(f"Error: {e}")


def demo_settings():
    """Demo TTS settings (rate, volume)."""
    print_section("3. TTS SETTINGS")

    from modules import TextToSpeech

    tts = TextToSpeech(voice="vi-female")
    text = "Đây là bài test tốc độ và âm lượng."

    settings = [
        {"rate": "+0%", "volume": "+0%", "desc": "Normal"},
        {"rate": "+30%", "volume": "+0%", "desc": "Fast"},
        {"rate": "-20%", "volume": "+0%", "desc": "Slow"},
        {"rate": "+0%", "volume": "+50%", "desc": "Loud"},
        {"rate": "+0%", "volume": "-30%", "desc": "Quiet"},
    ]

    print(f"\nText: {text}")

    for s in settings:
        print(f"\n--- {s['desc']} (rate={s['rate']}, volume={s['volume']}) ---")

        tts.set_rate(s['rate'])
        tts.set_volume(s['volume'])

        try:
            audio = tts.synthesize_sync(text)
            print(f"Audio size: {len(audio):,} bytes")
        except Exception as e:
            print(f"Error: {e}")


def demo_save_file(output_path: str = None):
    """Demo saving to file."""
    print_section("4. SAVE TO FILE")

    from modules import TextToSpeech
    import os

    if not output_path:
        output_path = "demo_tts_output.mp3"

    text = "Đây là file audio được tạo bởi hệ thống Text-to-Speech."

    print(f"\nText: {text}")
    print(f"Output: {output_path}")

    try:
        tts = TextToSpeech(voice="vi-female")
        tts.synthesize_to_file_sync(text, output_path)

        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"File created: {output_path}")
            print(f"File size: {size:,} bytes")

            # Cleanup demo file
            os.remove(output_path)
            print("(Demo file cleaned up)")
        else:
            print("File not created")

    except Exception as e:
        print(f"Error: {e}")


def demo_empty_text():
    """Demo handling empty text."""
    print_section("5. EDGE CASES")

    from modules import TextToSpeech

    tts = TextToSpeech()

    test_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("Hello", "Normal text"),
    ]

    for text, desc in test_cases:
        print(f"\n--- {desc} ---")
        print(f"Input: '{text}'")

        try:
            audio = tts.synthesize_sync(text)
            print(f"Audio size: {len(audio):,} bytes")
        except Exception as e:
            print(f"Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo TTS Module")
    parser.add_argument(
        "--text",
        type=str,
        help="Custom text to synthesize"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="vi-female",
        help="Voice to use"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save audio to file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("   TEXT-TO-SPEECH MODULE DEMO")
    print("=" * 60)

    if args.text:
        # Custom synthesis
        print_section("CUSTOM SYNTHESIS")
        from modules import TextToSpeech

        print(f"\nText: {args.text}")
        print(f"Voice: {args.voice}")

        tts = TextToSpeech(voice=args.voice)

        if args.save:
            tts.save_to_file(args.save, args.text)
            print(f"Saved to: {args.save}")
        else:
            audio = tts.synthesize_sync(args.text)
            print(f"Audio size: {len(audio):,} bytes")

    else:
        # Run all demos
        demo_voice_options()
        demo_synthesis()
        demo_settings()
        demo_save_file(args.save)
        demo_empty_text()

    print("\n" + "=" * 60)
    print("   DEMO COMPLETED!")
    print("=" * 60)
    print("\nUsage examples:")
    print("  python scripts/demo_tts.py --text 'Xin chao' --voice vi-female")
    print("  python scripts/demo_tts.py --text 'Hello' --voice en-male --save output.mp3")


if __name__ == "__main__":
    main()
