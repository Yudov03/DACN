# -*- coding: utf-8 -*-
"""
Test Post-Processing Cache
==========================

Comprehensive tests for PostProcessingCache functionality.

Run: python tests/test_cache.py
"""

import os
import sys
import time
import shutil
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.post_processing import PostProcessingCache, get_cache


class TestPostProcessingCache:
    """Test suite for PostProcessingCache."""

    def __init__(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_cache_")
        self.cache = PostProcessingCache(cache_dir=self.test_dir)
        self.passed = 0
        self.failed = 0
        self.results = []

    def cleanup(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def assert_equal(self, actual, expected, message=""):
        """Assert two values are equal."""
        if actual == expected:
            return True
        else:
            raise AssertionError(f"{message}\nExpected: {expected}\nActual: {actual}")

    def assert_none(self, value, message=""):
        """Assert value is None."""
        if value is None:
            return True
        else:
            raise AssertionError(f"{message}\nExpected: None\nActual: {value}")

    def assert_not_none(self, value, message=""):
        """Assert value is not None."""
        if value is not None:
            return True
        else:
            raise AssertionError(f"{message}\nExpected: not None\nActual: None")

    def assert_true(self, value, message=""):
        """Assert value is True."""
        if value:
            return True
        else:
            raise AssertionError(f"{message}\nExpected: True\nActual: {value}")

    def run_test(self, test_func, test_name):
        """Run a single test and track results."""
        try:
            test_func()
            self.passed += 1
            self.results.append((test_name, "PASS", None))
            print(f"  [PASS] {test_name}")
        except Exception as e:
            self.failed += 1
            self.results.append((test_name, "FAIL", str(e)))
            print(f"  [FAIL] {test_name}: {e}")

    # =========================================================================
    # Basic Operations
    # =========================================================================

    def test_set_and_get_basic(self):
        """Test basic set and get operations."""
        self.cache.clear()

        # Set a value
        self.cache.set("hello world", "ollama", "qwen2.5:7b", "xin chào thế giới")

        # Get it back
        result = self.cache.get("hello world", "ollama", "qwen2.5:7b")
        self.assert_equal(result, "xin chào thế giới", "Basic set/get failed")

    def test_cache_miss_nonexistent(self):
        """Test cache miss for nonexistent key."""
        self.cache.clear()

        result = self.cache.get("nonexistent text", "ollama", "qwen2.5:7b")
        self.assert_none(result, "Should return None for nonexistent key")

    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some entries
        self.cache.set("text1", "ollama", "model1", "result1")
        self.cache.set("text2", "ollama", "model1", "result2")

        # Clear
        self.cache.clear()

        # Should return None
        result1 = self.cache.get("text1", "ollama", "model1")
        result2 = self.cache.get("text2", "ollama", "model1")

        self.assert_none(result1, "text1 should be cleared")
        self.assert_none(result2, "text2 should be cleared")

    # =========================================================================
    # Cache Key Uniqueness
    # =========================================================================

    def test_different_text_different_key(self):
        """Test that different text produces different cache keys."""
        self.cache.clear()

        self.cache.set("text A", "ollama", "model", "result A")
        self.cache.set("text B", "ollama", "model", "result B")

        result_a = self.cache.get("text A", "ollama", "model")
        result_b = self.cache.get("text B", "ollama", "model")

        self.assert_equal(result_a, "result A", "text A should return result A")
        self.assert_equal(result_b, "result B", "text B should return result B")

    def test_different_method_different_key(self):
        """Test that different method produces different cache keys."""
        self.cache.clear()

        self.cache.set("same text", "ollama", "model", "ollama result")
        self.cache.set("same text", "transformer", "model", "transformer result")

        result_ollama = self.cache.get("same text", "ollama", "model")
        result_transformer = self.cache.get("same text", "transformer", "model")

        self.assert_equal(result_ollama, "ollama result")
        self.assert_equal(result_transformer, "transformer result")

    def test_different_model_different_key(self):
        """Test that different model produces different cache keys."""
        self.cache.clear()

        self.cache.set("same text", "ollama", "qwen2.5:7b", "qwen result")
        self.cache.set("same text", "ollama", "llama3.2", "llama result")

        result_qwen = self.cache.get("same text", "ollama", "qwen2.5:7b")
        result_llama = self.cache.get("same text", "ollama", "llama3.2")

        self.assert_equal(result_qwen, "qwen result")
        self.assert_equal(result_llama, "llama result")

    def test_substring_not_cached(self):
        """Test that substring of cached text is cache miss (exact match only)."""
        self.cache.clear()

        # Cache short text
        self.cache.set("Văn bằng", "ollama", "model", "Văn bằng (fixed)")

        # Try to get longer text containing the short text
        result = self.cache.get("Sinh viên nhận Văn bằng tốt nghiệp", "ollama", "model")
        self.assert_none(result, "Longer text should be cache miss")

        # Original short text should still work
        result_short = self.cache.get("Văn bằng", "ollama", "model")
        self.assert_equal(result_short, "Văn bằng (fixed)")

    def test_superstring_not_cached(self):
        """Test that superstring of cached text is cache miss."""
        self.cache.clear()

        # Cache long text
        self.cache.set("Sinh viên nhận Văn bằng tốt nghiệp", "ollama", "model", "Long result")

        # Try to get shorter text
        result = self.cache.get("Văn bằng", "ollama", "model")
        self.assert_none(result, "Shorter text should be cache miss")

    # =========================================================================
    # Vietnamese Text
    # =========================================================================

    def test_vietnamese_text(self):
        """Test caching Vietnamese text with diacritics."""
        self.cache.clear()

        input_text = "Đại học Quốc gia Thành phố Hồ Chí Minh"
        output_text = "Đại học Quốc gia Thành phố Hồ Chí Minh (verified)"

        self.cache.set(input_text, "ollama", "qwen2.5:7b", output_text)
        result = self.cache.get(input_text, "ollama", "qwen2.5:7b")

        self.assert_equal(result, output_text, "Vietnamese text should be cached correctly")

    def test_vietnamese_ocr_errors(self):
        """Test caching text with OCR errors."""
        self.cache.clear()

        ocr_input = "ĐI HC QUC GIA"
        corrected = "ĐẠI HỌC QUỐC GIA"

        self.cache.set(ocr_input, "ollama", "qwen2.5:7b", corrected)
        result = self.cache.get(ocr_input, "ollama", "qwen2.5:7b")

        self.assert_equal(result, corrected)

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_empty_text(self):
        """Test handling of empty text."""
        self.cache.clear()

        self.cache.set("", "ollama", "model", "")
        result = self.cache.get("", "ollama", "model")

        self.assert_equal(result, "", "Empty text should be cached")

    def test_whitespace_only(self):
        """Test handling of whitespace-only text."""
        self.cache.clear()

        self.cache.set("   \n\t  ", "ollama", "model", "   \n\t  ")
        result = self.cache.get("   \n\t  ", "ollama", "model")

        self.assert_equal(result, "   \n\t  ")

    def test_long_text(self):
        """Test caching very long text."""
        self.cache.clear()

        # Create 10KB of text
        long_input = "Đây là một đoạn văn bản rất dài. " * 500
        long_output = "Đây là kết quả đã được xử lý. " * 500

        self.cache.set(long_input, "ollama", "qwen2.5:7b", long_output)
        result = self.cache.get(long_input, "ollama", "qwen2.5:7b")

        self.assert_equal(result, long_output, "Long text should be cached correctly")

    def test_special_characters(self):
        """Test text with special characters."""
        self.cache.clear()

        special_text = "Text với các ký tự đặc biệt: !@#$%^&*()[]{}|\\:\";<>?,./"
        self.cache.set(special_text, "ollama", "model", "processed")
        result = self.cache.get(special_text, "ollama", "model")

        self.assert_equal(result, "processed")

    def test_multiline_text(self):
        """Test multiline text."""
        self.cache.clear()

        multiline = """Dòng 1
Dòng 2
Dòng 3 với tiếng Việt
"""
        self.cache.set(multiline, "ollama", "model", "processed multiline")
        result = self.cache.get(multiline, "ollama", "model")

        self.assert_equal(result, "processed multiline")

    # =========================================================================
    # Index Integrity
    # =========================================================================

    def test_index_file_created(self):
        """Test that index file is created."""
        self.cache.clear()
        self.cache.set("test", "ollama", "model", "result")

        self.assert_true(
            os.path.exists(self.cache.index_file),
            "Index file should be created"
        )

    def test_cache_file_created(self):
        """Test that cache file is created."""
        self.cache.clear()
        self.cache.set("test", "ollama", "model", "result")

        # Get the cache key
        key = self.cache.get_cache_key("test", "ollama", "model")
        cache_file = self.cache.chunks_dir / f"{key}.txt"

        self.assert_true(
            os.path.exists(cache_file),
            f"Cache file should be created at {cache_file}"
        )

    def test_index_out_of_sync(self):
        """Test handling when index is out of sync with files."""
        self.cache.clear()

        # Set a value
        self.cache.set("test", "ollama", "model", "result")

        # Delete the cache file manually (simulating corruption)
        key = self.cache.get_cache_key("test", "ollama", "model")
        cache_file = self.cache.chunks_dir / f"{key}.txt"
        os.remove(cache_file)

        # Should return None and clean up index
        result = self.cache.get("test", "ollama", "model")
        self.assert_none(result, "Should return None when file is missing")

    # =========================================================================
    # Statistics
    # =========================================================================

    def test_stats_empty(self):
        """Test stats on empty cache."""
        self.cache.clear()
        stats = self.cache.get_stats()

        self.assert_equal(stats["entries"], 0, "Empty cache should have 0 entries")
        self.assert_equal(stats["total_size_bytes"], 0, "Empty cache should have 0 bytes")

    def test_stats_with_entries(self):
        """Test stats with cached entries."""
        self.cache.clear()

        self.cache.set("text1", "ollama", "model", "result1")
        self.cache.set("text2", "ollama", "model", "result2")
        self.cache.set("text3", "ollama", "model", "result3")

        stats = self.cache.get_stats()

        self.assert_equal(stats["entries"], 3, "Should have 3 entries")
        self.assert_true(stats["total_size_bytes"] > 0, "Should have non-zero size")

    # =========================================================================
    # Persistence
    # =========================================================================

    def test_persistence_across_instances(self):
        """Test that cache persists across different cache instances."""
        self.cache.clear()

        # Set value with first instance
        self.cache.set("persistent text", "ollama", "model", "persistent result")

        # Create new instance with same directory
        cache2 = PostProcessingCache(cache_dir=self.test_dir)

        # Should get the cached value
        result = cache2.get("persistent text", "ollama", "model")
        self.assert_equal(result, "persistent result", "Cache should persist")

    # =========================================================================
    # Cache Key Generation
    # =========================================================================

    def test_cache_key_deterministic(self):
        """Test that same input produces same cache key."""
        key1 = self.cache.get_cache_key("text", "ollama", "model")
        key2 = self.cache.get_cache_key("text", "ollama", "model")

        self.assert_equal(key1, key2, "Same input should produce same key")

    def test_cache_key_length(self):
        """Test that cache key is MD5 hash (32 chars)."""
        key = self.cache.get_cache_key("text", "ollama", "model")

        self.assert_equal(len(key), 32, "MD5 hash should be 32 characters")

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all(self):
        """Run all tests."""
        print("\n" + "=" * 60)
        print("POST-PROCESSING CACHE TESTS")
        print("=" * 60)

        tests = [
            # Basic Operations
            (self.test_set_and_get_basic, "Basic set and get"),
            (self.test_cache_miss_nonexistent, "Cache miss for nonexistent"),
            (self.test_clear_cache, "Clear cache"),

            # Cache Key Uniqueness
            (self.test_different_text_different_key, "Different text = different key"),
            (self.test_different_method_different_key, "Different method = different key"),
            (self.test_different_model_different_key, "Different model = different key"),
            (self.test_substring_not_cached, "Substring not cached (exact match)"),
            (self.test_superstring_not_cached, "Superstring not cached"),

            # Vietnamese Text
            (self.test_vietnamese_text, "Vietnamese text with diacritics"),
            (self.test_vietnamese_ocr_errors, "Vietnamese OCR errors"),

            # Edge Cases
            (self.test_empty_text, "Empty text"),
            (self.test_whitespace_only, "Whitespace-only text"),
            (self.test_long_text, "Long text (10KB)"),
            (self.test_special_characters, "Special characters"),
            (self.test_multiline_text, "Multiline text"),

            # Index Integrity
            (self.test_index_file_created, "Index file created"),
            (self.test_cache_file_created, "Cache file created"),
            (self.test_index_out_of_sync, "Index out of sync handling"),

            # Statistics
            (self.test_stats_empty, "Stats on empty cache"),
            (self.test_stats_with_entries, "Stats with entries"),

            # Persistence
            (self.test_persistence_across_instances, "Persistence across instances"),

            # Cache Key
            (self.test_cache_key_deterministic, "Cache key deterministic"),
            (self.test_cache_key_length, "Cache key length (MD5)"),
        ]

        print(f"\nRunning {len(tests)} tests...\n")

        for test_func, test_name in tests:
            self.run_test(test_func, test_name)

        # Summary
        print("\n" + "-" * 60)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("-" * 60)

        if self.failed > 0:
            print("\nFailed tests:")
            for name, status, error in self.results:
                if status == "FAIL":
                    print(f"  - {name}: {error}")

        # Cleanup
        self.cleanup()

        return self.failed == 0


def test_get_cache_function():
    """Test the get_cache() function."""
    print("\n" + "=" * 60)
    print("TEST: get_cache() function")
    print("=" * 60)

    # Save original env
    original = os.environ.get("POSTPROCESS_CACHE")

    # Test with cache enabled
    os.environ["POSTPROCESS_CACHE"] = "true"
    # Reset global cache instance
    import src.modules.post_processing as pp
    pp._cache_instance = None

    cache = get_cache()
    if cache is not None:
        print("  [PASS] get_cache() returns cache when enabled")
    else:
        print("  [FAIL] get_cache() returned None when enabled")
        return False

    # Test with cache disabled
    os.environ["POSTPROCESS_CACHE"] = "false"
    pp._cache_instance = None

    cache = get_cache()
    if cache is None:
        print("  [PASS] get_cache() returns None when disabled")
    else:
        print("  [FAIL] get_cache() returned cache when disabled")
        return False

    # Restore
    if original:
        os.environ["POSTPROCESS_CACHE"] = original
    else:
        os.environ.pop("POSTPROCESS_CACHE", None)
    pp._cache_instance = None

    return True


def test_cache_performance():
    """Test cache performance (HIT should be much faster than MISS)."""
    print("\n" + "=" * 60)
    print("TEST: Cache Performance")
    print("=" * 60)

    cache_dir = tempfile.mkdtemp(prefix="test_perf_")
    cache = PostProcessingCache(cache_dir=cache_dir)

    # Large text
    large_text = "Đây là văn bản tiếng Việt cần xử lý. " * 100

    # Measure set time
    start = time.time()
    cache.set(large_text, "ollama", "model", "processed " * 100)
    set_time = time.time() - start

    # Measure get time (cache hit)
    start = time.time()
    for _ in range(100):
        result = cache.get(large_text, "ollama", "model")
    get_time = (time.time() - start) / 100

    # Measure miss time
    start = time.time()
    for _ in range(100):
        result = cache.get("nonexistent " * 50, "ollama", "model")
    miss_time = (time.time() - start) / 100

    print(f"  Set time: {set_time*1000:.2f}ms")
    print(f"  Get time (HIT): {get_time*1000:.2f}ms")
    print(f"  Get time (MISS): {miss_time*1000:.2f}ms")

    # Cleanup
    shutil.rmtree(cache_dir)

    if get_time < 0.01:  # Less than 10ms
        print("  [PASS] Cache HIT is fast (<10ms)")
        return True
    else:
        print("  [FAIL] Cache HIT is slow (>10ms)")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("POST-PROCESSING CACHE TEST SUITE")
    print("=" * 60)

    all_passed = True

    # Run main test suite
    test_suite = TestPostProcessingCache()
    if not test_suite.run_all():
        all_passed = False

    # Run get_cache function test
    if not test_get_cache_function():
        all_passed = False

    # Run performance test
    if not test_cache_performance():
        all_passed = False

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
