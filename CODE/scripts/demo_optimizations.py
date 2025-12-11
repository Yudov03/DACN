"""
Demo script for Phase 1 Optimizations
- Query Expansion
- Context Compression
- Caching
- Better Prompts
"""

import sys
import os
import io
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_query_expansion():
    """Demo Query Expansion module"""
    print("\n" + "=" * 60)
    print("1. QUERY EXPANSION DEMO")
    print("=" * 60)

    from modules.query_expansion_module import QueryExpander

    expander = QueryExpander(method="synonym", num_expansions=3)

    test_queries = [
        "AI la gi?",
        "Machine learning hoat dong nhu the nao?",
        "Tai sao deep learning quan trong?",
        "What is NLP?",
    ]

    print("\n--- Synonym Expansion ---")
    for query in test_queries:
        expanded = expander.expand(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded ({len(expanded)}): {expanded}")


def demo_context_compression():
    """Demo Context Compression module"""
    print("\n" + "=" * 60)
    print("2. CONTEXT COMPRESSION DEMO")
    print("=" * 60)

    from modules.context_compression_module import ContextCompressor

    compressor = ContextCompressor(method="extractive", max_tokens=500)

    test_contexts = [
        {
            "text": "Machine learning la mot nhanh cua tri tue nhan tao. No cho phep may tinh hoc tu du lieu. Cac ung dung bao gom nhan dien hinh anh, xu ly ngon ngu tu nhien. Deep learning la mot dang dac biet cua machine learning. Hom nay troi dep qua. Toi di an sang. Cau nay khong lien quan.",
            "metadata": {"start_time": 0.0, "end_time": 30.0, "audio_file": "lecture.mp3"}
        },
        {
            "text": "NLP giup may tinh hieu ngon ngu con nguoi. Cac mo hinh nhu BERT va GPT da tao ra dot pha. Transformer la kien truc quan trong trong NLP. An sang rat ngon. Toi thich uong tra.",
            "metadata": {"start_time": 30.0, "end_time": 60.0, "audio_file": "lecture.mp3"}
        }
    ]

    query = "Machine learning la gi?"

    print(f"\nQuery: {query}")
    print(f"Original contexts: {len(test_contexts)}")

    context_str, compressed = compressor.compress(query, test_contexts)

    print(f"\nCompressed context:\n{context_str}")

    for ctx in compressed:
        orig = ctx.get("original_length", len(ctx["text"]))
        comp = ctx.get("compressed_length", len(ctx["text"]))
        ratio = comp / orig * 100 if orig > 0 else 0
        print(f"  - {orig} -> {comp} chars ({ratio:.1f}%)")


def demo_caching():
    """Demo Caching module"""
    print("\n" + "=" * 60)
    print("3. CACHING DEMO")
    print("=" * 60)

    from modules.caching_module import CacheManager, LRUCache
    import numpy as np

    # Test LRU Cache
    print("\n--- LRU Cache ---")
    cache = LRUCache(max_size=3, ttl=10)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    print(f"Get key1: {cache.get('key1')}")
    print(f"Get key2: {cache.get('key2')}")
    print(f"Stats: {cache.get_stats()}")

    # Add 4th item (should evict oldest)
    cache.set("key4", "value4")
    print(f"After adding key4, get key3: {cache.get('key3')}")

    # Test Cache Manager
    print("\n--- Cache Manager ---")
    manager = CacheManager(cache_dir="./test_cache_demo")

    # Simulate embedding cache
    test_embedding = np.random.randn(768).tolist()
    manager.set_embedding("Test text for embedding", "test-model", test_embedding)

    cached = manager.get_embedding("Test text for embedding", "test-model")
    print(f"Cached embedding length: {len(cached) if cached else 0}")

    # Test cache hit
    start = time.time()
    for _ in range(100):
        manager.get_embedding("Test text for embedding", "test-model")
    elapsed = (time.time() - start) * 1000
    print(f"100 cache hits: {elapsed:.2f}ms ({elapsed/100:.3f}ms per hit)")

    manager.print_stats()

    # Cleanup
    import shutil
    shutil.rmtree("./test_cache_demo", ignore_errors=True)


def demo_prompt_templates():
    """Demo Prompt Templates module"""
    print("\n" + "=" * 60)
    print("4. PROMPT TEMPLATES DEMO")
    print("=" * 60)

    from modules.prompt_templates import PromptTemplateManager, get_rag_prompt

    # Vietnamese
    print("\n--- Vietnamese Templates ---")
    manager_vi = PromptTemplateManager(language="vi")
    print(f"Available templates: {manager_vi.list_templates()}")

    context = "[00:15-00:30] Machine learning la mot nhanh cua AI. No cho phep may tinh hoc tu du lieu."
    question = "Machine learning la gi?"

    for template_name in manager_vi.list_templates():
        sys_prompt, user_prompt = manager_vi.format_prompt(
            template_name,
            context=context,
            question=question
        )
        print(f"\n{template_name}:")
        print(f"  System: {sys_prompt[:60]}...")
        print(f"  User: {user_prompt[:80]}...")

    # Quick function
    print("\n--- Quick Function ---")
    sys_p, user_p = get_rag_prompt(
        context="Sample context",
        question="Sample question",
        template="audio_qa",
        language="vi"
    )
    print(f"System: {sys_p}")
    print(f"User: {user_p[:50]}...")


def demo_integrated():
    """Demo tich hop tat ca optimizations"""
    print("\n" + "=" * 60)
    print("5. INTEGRATED DEMO")
    print("=" * 60)

    from modules.query_expansion_module import QueryExpander
    from modules.context_compression_module import ContextCompressor
    from modules.caching_module import CacheManager
    from modules.prompt_templates import PromptTemplateManager

    print("\nSimulating optimized RAG pipeline...")

    # 1. Setup
    expander = QueryExpander(method="synonym")
    compressor = ContextCompressor(method="extractive", max_tokens=500)
    cache = CacheManager(cache_dir="./test_cache_integrated")
    prompts = PromptTemplateManager(language="vi")

    # 2. Input
    query = "Machine learning la gi?"
    contexts = [
        {
            "text": "Machine learning la mot nhanh cua AI. No cho phep may tinh hoc tu du lieu ma khong can lap trinh tuong minh.",
            "metadata": {"start_time": 0.0, "end_time": 15.0}
        },
        {
            "text": "Deep learning su dung neural networks voi nhieu layers. Day la dang tien tien cua machine learning.",
            "metadata": {"start_time": 15.0, "end_time": 30.0}
        }
    ]

    # 3. Query Expansion
    print(f"\nOriginal query: {query}")
    expanded_queries = expander.expand(query)
    print(f"Expanded queries: {expanded_queries}")

    # 4. Context Compression
    compressed_context, _ = compressor.compress(query, contexts)
    print(f"\nCompressed context:\n{compressed_context}")

    # 5. Format Prompt
    sys_prompt, user_prompt = prompts.format_prompt(
        "audio_qa",
        context=compressed_context,
        question=query
    )
    print(f"\nFormatted prompt:")
    print(f"  System: {sys_prompt}")
    print(f"  User:\n{user_prompt}")

    # 6. Cache stats
    cache.print_stats()

    # Cleanup
    import shutil
    shutil.rmtree("./test_cache_integrated", ignore_errors=True)


def main():
    print("=" * 60)
    print("PHASE 1 OPTIMIZATIONS DEMO")
    print("=" * 60)

    # Run all demos
    demo_query_expansion()
    demo_context_compression()
    demo_caching()
    demo_prompt_templates()
    demo_integrated()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNew modules added:")
    print("  - query_expansion_module.py: QueryExpander, MultiQueryRetriever")
    print("  - context_compression_module.py: ContextCompressor, ContextualCompressor")
    print("  - caching_module.py: CacheManager, EmbeddingCache, ResponseCache")
    print("  - prompt_templates.py: PromptTemplateManager, get_rag_prompt")


if __name__ == "__main__":
    main()
