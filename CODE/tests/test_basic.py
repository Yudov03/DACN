"""
Basic tests - Khong can dependencies phuc tap
Test cac modules core logic
"""

import sys
from pathlib import Path
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_chunking_basic():
    """Test Chunking Module - không cần dependencies"""
    print("\n" + "=" * 80)
    print("TEST: CHUNKING MODULE (Basic)")
    print("=" * 80)

    try:
        from modules.chunking_module import TextChunker

        # Test 1: Fixed size chunking
        print("\n[Test 1] Fixed size chunking...")
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, method="fixed")

        sample_text = """Trí tuệ nhân tạo (AI) đang thay đổi thế giới. Công nghệ này có ứng dụng trong nhiều lĩnh vực như y tế, giáo dục, và kinh doanh. Machine Learning là một nhánh quan trọng của AI. Nó cho phép máy tính học từ dữ liệu mà không cần được lập trình cụ thể. Deep Learning sử dụng neural networks nhiều lớp."""

        chunks = chunker.chunk_text(sample_text.strip())
        print(f"✓ Created {len(chunks)} chunks")
        assert len(chunks) > 0, "No chunks created!"

        for i, chunk in enumerate(chunks[:3]):
            print(f"  Chunk {i}: {chunk['word_count']} words")
            assert 'text' in chunk, "Missing text field!"
            assert 'chunk_id' in chunk, "Missing chunk_id!"
            assert 'word_count' in chunk, "Missing word_count!"

        # Test 2: Sentence chunking
        print("\n[Test 2] Sentence-based chunking...")
        chunker_sentence = TextChunker(chunk_size=150, method="sentence")
        chunks_sentence = chunker_sentence.chunk_text(sample_text.strip())
        print(f"✓ Created {len(chunks_sentence)} chunks")

        # Test 3: Semantic chunking
        print("\n[Test 3] Semantic chunking...")
        chunker_semantic = TextChunker(chunk_size=200, method="semantic")
        chunks_semantic = chunker_semantic.chunk_text(sample_text.strip())
        print(f"✓ Created {len(chunks_semantic)} chunks")

        # Test 4: Metadata
        print("\n[Test 4] Chunking with metadata...")
        metadata = {"source": "test", "audio_file": "test.mp3"}
        chunks_meta = chunker.chunk_text(sample_text, metadata=metadata)
        print(f"✓ Created {len(chunks_meta)} chunks with metadata")
        assert chunks_meta[0]['source'] == "test", "Metadata not preserved!"

        print("\n✅ CHUNKING MODULE: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test Config Module"""
    print("\n" + "=" * 80)
    print("TEST: CONFIG MODULE")
    print("=" * 80)

    try:
        from config import Config

        print("\n[Test 1] Check config values...")
        assert Config.CHUNK_SIZE > 0, "Invalid CHUNK_SIZE!"
        assert Config.CHUNK_OVERLAP >= 0, "Invalid CHUNK_OVERLAP!"
        assert Config.TOP_K > 0, "Invalid TOP_K!"
        print(f"✓ CHUNK_SIZE: {Config.CHUNK_SIZE}")
        print(f"✓ CHUNK_OVERLAP: {Config.CHUNK_OVERLAP}")
        print(f"✓ TOP_K: {Config.TOP_K}")

        print("\n[Test 2] Check paths...")
        assert Config.DATA_DIR.exists(), "DATA_DIR not found!"
        assert Config.AUDIO_DIR.exists(), "AUDIO_DIR not found!"
        print(f"✓ DATA_DIR: {Config.DATA_DIR}")
        print(f"✓ AUDIO_DIR: {Config.AUDIO_DIR}")

        print("\n[Test 3] Validate config...")
        result = Config.validate()
        assert result == True, "Config validation failed!"
        print("✓ Config validation passed")

        print("\n✅ CONFIG MODULE: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test module imports"""
    print("\n" + "=" * 80)
    print("TEST: MODULE IMPORTS")
    print("=" * 80)

    modules_to_test = [
        ('config', 'Config'),
        ('modules.chunking_module', 'TextChunker'),
    ]

    passed = 0
    failed = 0

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {str(e)}")
            failed += 1

    print(f"\n{passed}/{len(modules_to_test)} imports successful")

    if failed == 0:
        print("\n✅ ALL IMPORTS PASSED")
        return True
    else:
        print(f"\n⚠️ {failed} import(s) failed")
        return False


def run_basic_tests():
    """Chạy basic tests"""
    print("\n" + "=" * 80)
    print("RUNNING BASIC TESTS (No heavy dependencies required)")
    print("=" * 80)

    results = {
        'imports': test_imports(),
        'config': test_config(),
        'chunking': test_chunking_basic()
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL BASIC TESTS PASSED!")
        print("\nNote: Để test đầy đủ, cần cài đặt dependencies:")
        print("  pip install -r requirements.txt")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
