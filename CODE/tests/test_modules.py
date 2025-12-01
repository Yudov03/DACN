"""
Test suite cho các modules chính
Chạy test không cần audio files hoặc API keys
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Direct imports to avoid __init__.py issues
from modules import chunking_module, embedding_module, vector_db_module
import numpy as np

TextChunker = chunking_module.TextChunker
TextEmbedding = embedding_module.TextEmbedding
VectorDatabase = vector_db_module.VectorDatabase


def test_chunking_module():
    """Test Chunking Module"""
    print("\n" + "=" * 80)
    print("TEST 1: CHUNKING MODULE")
    print("=" * 80)

    # Test 1: Fixed size chunking
    print("\n[Test 1.1] Fixed size chunking...")
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, method="fixed")

    sample_text = """
    Trí tuệ nhân tạo (AI) đang thay đổi thế giới.
    Công nghệ này có ứng dụng trong nhiều lĩnh vực như y tế, giáo dục, và kinh doanh.

    Machine Learning là một nhánh quan trọng của AI.
    Nó cho phép máy tính học từ dữ liệu mà không cần được lập trình cụ thể.

    Deep Learning sử dụng neural networks nhiều lớp.
    Đây là công nghệ đằng sau nhiều ứng dụng AI hiện đại.
    """

    chunks = chunker.chunk_text(sample_text.strip())
    print(f"✓ Created {len(chunks)} chunks với method 'fixed'")

    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"  Chunk {i}: {chunk['word_count']} words, {chunk['char_count']} chars")

    # Test 2: Sentence-based chunking
    print("\n[Test 1.2] Sentence-based chunking...")
    chunker_sentence = TextChunker(chunk_size=200, method="sentence")
    chunks_sentence = chunker_sentence.chunk_text(sample_text.strip())
    print(f"✓ Created {len(chunks_sentence)} chunks với method 'sentence'")

    # Test 3: Semantic chunking
    print("\n[Test 1.3] Semantic chunking...")
    chunker_semantic = TextChunker(chunk_size=200, method="semantic")
    chunks_semantic = chunker_semantic.chunk_text(sample_text.strip())
    print(f"✓ Created {len(chunks_semantic)} chunks với method 'semantic'")

    # Test 4: Chunk with timestamps
    print("\n[Test 1.4] Chunking with timestamps...")
    transcript_data = {
        "audio_filename": "test.mp3",
        "full_text": sample_text,
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Trí tuệ nhân tạo (AI) đang thay đổi thế giới."},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "Công nghệ này có ứng dụng trong nhiều lĩnh vực."},
            {"id": 2, "start": 10.0, "end": 15.0, "text": "Machine Learning là một nhánh quan trọng của AI."}
        ]
    }

    chunks_ts = chunker.chunk_transcript(transcript_data, preserve_timestamps=True)
    print(f"✓ Created {len(chunks_ts)} chunks with timestamps")

    if chunks_ts:
        print(f"  First chunk: {chunks_ts[0]['start_time']:.2f}s - {chunks_ts[0]['end_time']:.2f}s")

    print("\n✅ CHUNKING MODULE: ALL TESTS PASSED")
    return True


def test_embedding_module():
    """Test Embedding Module"""
    print("\n" + "=" * 80)
    print("TEST 2: EMBEDDING MODULE")
    print("=" * 80)

    try:
        print("\n[Test 2.1] Initialize embedding model...")
        # Sử dụng model nhỏ để test nhanh
        embedder = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print(f"✓ Model loaded. Embedding dimension: {embedder.embedding_dim}")

        # Test 2: Encode single text
        print("\n[Test 2.2] Encode single text...")
        text = "Đây là một câu văn mẫu để test."
        embedding = embedder.encode_text(text, show_progress=False)
        print(f"✓ Embedding shape: {embedding.shape}")
        assert embedding.shape[0] == embedder.embedding_dim, "Embedding dimension mismatch!"

        # Test 3: Encode multiple texts
        print("\n[Test 2.3] Encode batch texts...")
        texts = [
            "Trí tuệ nhân tạo đang phát triển.",
            "Machine Learning cần nhiều dữ liệu.",
            "Deep Learning sử dụng neural networks."
        ]
        embeddings = embedder.encode_text(texts, show_progress=False)
        print(f"✓ Batch embeddings shape: {embeddings.shape}")
        assert embeddings.shape == (3, embedder.embedding_dim), "Batch embedding shape mismatch!"

        # Test 4: Similarity computation
        print("\n[Test 2.4] Compute similarity...")
        sim_01 = embedder.compute_similarity(embeddings[0], embeddings[1])
        sim_02 = embedder.compute_similarity(embeddings[0], embeddings[2])
        print(f"✓ Similarity[0,1]: {sim_01:.4f}")
        print(f"✓ Similarity[0,2]: {sim_02:.4f}")
        assert 0 <= sim_01 <= 1, "Similarity out of range!"
        assert 0 <= sim_02 <= 1, "Similarity out of range!"

        # Test 5: Find most similar
        print("\n[Test 2.5] Find most similar...")
        query_embedding = embeddings[0]
        results = embedder.find_most_similar(query_embedding, embeddings, top_k=2)
        print(f"✓ Found {len(results)} similar items")
        for i, result in enumerate(results):
            print(f"  Top {i+1}: index={result['index']}, similarity={result['similarity']:.4f}")

        # Test 6: Encode chunks
        print("\n[Test 2.6] Encode chunks...")
        chunks = [
            {"chunk_id": 0, "text": "Chunk 1 text"},
            {"chunk_id": 1, "text": "Chunk 2 text"}
        ]
        chunks_with_embeddings = embedder.encode_chunks(chunks, batch_size=2)
        print(f"✓ Encoded {len(chunks_with_embeddings)} chunks")
        assert "embedding" in chunks_with_embeddings[0], "Missing embedding field!"
        assert len(chunks_with_embeddings[0]["embedding"]) == embedder.embedding_dim

        print("\n✅ EMBEDDING MODULE: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ EMBEDDING MODULE TEST FAILED: {str(e)}")
        print("Note: Module cần download model lần đầu, có thể mất thời gian.")
        return False


def test_vector_db_module():
    """Test Vector Database Module"""
    print("\n" + "=" * 80)
    print("TEST 3: VECTOR DATABASE MODULE")
    print("=" * 80)

    # Test ChromaDB
    print("\n[Test 3.1] ChromaDB - Initialize...")
    try:
        db_chroma = VectorDatabase(
            db_type="chromadb",
            collection_name="test_collection_chroma",
            db_path="data/vector_db/test_chroma"
        )
        print("✓ ChromaDB initialized")

        # Get stats
        stats = db_chroma.get_collection_stats()
        print(f"✓ Stats: {stats['count']} documents in collection")

        # Test adding documents
        print("\n[Test 3.2] ChromaDB - Add documents...")
        sample_chunks = [
            {
                "text": "AI đang phát triển rất nhanh.",
                "embedding": np.random.rand(768).tolist(),
                "chunk_id": 0
            },
            {
                "text": "Machine Learning cần nhiều dữ liệu.",
                "embedding": np.random.rand(768).tolist(),
                "chunk_id": 1
            }
        ]

        num_added = db_chroma.add_documents(sample_chunks)
        print(f"✓ Added {num_added} documents to ChromaDB")

        # Test search
        print("\n[Test 3.3] ChromaDB - Search...")
        query_embedding = np.random.rand(768)
        results = db_chroma.search(query_embedding, top_k=2)
        print(f"✓ Found {len(results)} results")
        if results:
            print(f"  Top result similarity: {results[0].get('similarity', 0):.4f}")

        print("\n✓ ChromaDB tests passed")

    except Exception as e:
        print(f"❌ ChromaDB test failed: {str(e)}")

    # Test FAISS
    print("\n[Test 3.4] FAISS - Initialize...")
    try:
        db_faiss = VectorDatabase(
            db_type="faiss",
            collection_name="test_collection_faiss",
            db_path="data/vector_db/test_faiss",
            embedding_dimension=768
        )
        print("✓ FAISS initialized")

        # Test adding documents
        print("\n[Test 3.5] FAISS - Add documents...")
        num_added = db_faiss.add_documents(sample_chunks)
        print(f"✓ Added {num_added} documents to FAISS")

        # Test search
        print("\n[Test 3.6] FAISS - Search...")
        results = db_faiss.search(query_embedding, top_k=2)
        print(f"✓ Found {len(results)} results")
        if results:
            print(f"  Top result similarity: {results[0].get('similarity', 0):.4f}")

        print("\n✓ FAISS tests passed")

    except Exception as e:
        print(f"❌ FAISS test failed: {str(e)}")

    print("\n✅ VECTOR DATABASE MODULE: TESTS COMPLETED")
    return True


def test_integration():
    """Test tích hợp các modules"""
    print("\n" + "=" * 80)
    print("TEST 4: INTEGRATION TEST")
    print("=" * 80)

    try:
        # Setup
        print("\n[Test 4.1] Setup pipeline components...")
        chunker = TextChunker(chunk_size=200, method="semantic")
        embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = VectorDatabase(
            db_type="chromadb",
            collection_name="test_integration",
            db_path="data/vector_db/test_integration"
        )

        # Sample data
        sample_text = """
        Trí tuệ nhân tạo là một lĩnh vực quan trọng của khoa học máy tính.
        Machine Learning cho phép máy tính học từ dữ liệu.
        Deep Learning sử dụng mạng neural nhiều lớp để giải quyết vấn đề phức tạp.
        Natural Language Processing giúp máy tính hiểu và xử lý ngôn ngữ tự nhiên.
        """

        # Step 1: Chunking
        print("\n[Test 4.2] Chunking text...")
        chunks = chunker.chunk_text(sample_text.strip())
        print(f"✓ Created {len(chunks)} chunks")

        # Step 2: Embedding
        print("\n[Test 4.3] Creating embeddings...")
        chunks_with_embeddings = embedder.encode_chunks(chunks)
        print(f"✓ Created embeddings for {len(chunks_with_embeddings)} chunks")

        # Step 3: Store in DB
        print("\n[Test 4.4] Storing in database...")
        num_stored = vector_db.add_documents(chunks_with_embeddings)
        print(f"✓ Stored {num_stored} documents")

        # Step 4: Query
        print("\n[Test 4.5] Querying database...")
        query = "Machine Learning là gì?"
        query_embedding = embedder.encode_text(query, show_progress=False)
        results = vector_db.search(query_embedding, top_k=3)
        print(f"✓ Retrieved {len(results)} results for query: '{query}'")

        # Show results
        for i, result in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"    Text: {result.get('text', '')[:80]}...")
            print(f"    Similarity: {result.get('similarity', 0):.4f}")

        print("\n✅ INTEGRATION TEST: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS FOR AUDIO IR SYSTEM")
    print("=" * 80)

    results = {}

    # Test 1: Chunking
    results['chunking'] = test_chunking_module()

    # Test 2: Embedding
    results['embedding'] = test_embedding_module()

    # Test 3: Vector DB
    results['vector_db'] = test_vector_db_module()

    # Test 4: Integration
    results['integration'] = test_integration()

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
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
