"""
Test file cho cac modules moi (Qdrant + OpenAI/Google Embeddings + LangChain)
"""

import sys
import os
import io
from pathlib import Path

# Fix Windows encoding for Vietnamese characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env with override to ensure latest values
from dotenv import load_dotenv
load_dotenv(override=True)

def test_config():
    """Test config module"""
    print("\n" + "=" * 60)
    print("TEST 1: Config Module")
    print("=" * 60)

    try:
        from config import Config

        print(f"BASE_DIR: {Config.BASE_DIR}")
        print(f"EMBEDDING_MODEL: {Config.EMBEDDING_MODEL}")
        print(f"EMBEDDING_DIMENSION: {Config.EMBEDDING_DIMENSION}")
        print(f"LLM_MODEL: {Config.LLM_MODEL}")
        print(f"QDRANT_HOST: {Config.QDRANT_HOST}")
        print(f"QDRANT_PORT: {Config.QDRANT_PORT}")
        print(f"COLLECTION_NAME: {Config.COLLECTION_NAME}")
        print(f"CHUNK_SIZE: {Config.CHUNK_SIZE}")
        print(f"CHUNKING_METHOD: {Config.CHUNKING_METHOD}")

        # Check API key
        if Config.OPENAI_API_KEY:
            print(f"OPENAI_API_KEY: {'*' * 10}...{Config.OPENAI_API_KEY[-4:]}")
        else:
            print("WARNING: OPENAI_API_KEY not set!")

        print("\n[PASS] Config module works correctly!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Config error: {e}")
        return False


def test_chunking_basic():
    """Test chunking module (basic - khong can API)"""
    print("\n" + "=" * 60)
    print("TEST 2: Chunking Module (Basic)")
    print("=" * 60)

    try:
        from modules.chunking_module import TextChunker

        # Test voi method='recursive' (khong can API)
        chunker = TextChunker(
            chunk_size=200,
            chunk_overlap=20,
            method="recursive"
        )

        sample_text = """
        Day la doan van ban mau de test chunking.
        No co nhieu cau va nhieu paragraph.

        Paragraph thu hai noi ve AI va Machine Learning.
        Cac mo hinh hoc sau dang phat trien nhanh.

        Paragraph thu ba la ket luan.
        He thong hoat dong tot.
        """

        chunks = chunker.chunk_text(sample_text)
        print(f"So chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i}:")
            print(f"  Text: {chunk['text'][:50]}...")
            print(f"  Chars: {chunk['char_count']}")

        # Test transcript chunking
        sample_transcript = {
            "audio_filename": "test.mp3",
            "full_text": "Day la transcript.",
            "segments": [
                {"id": 0, "text": "Cau thu nhat.", "start": 0.0, "end": 2.5},
                {"id": 1, "text": "Cau thu hai.", "start": 2.5, "end": 5.0},
            ]
        }

        transcript_chunks = chunker.chunk_transcript(sample_transcript)
        print(f"\nTranscript chunks: {len(transcript_chunks)}")

        print("\n[PASS] Chunking module (basic) works correctly!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Chunking error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_db_inmemory():
    """Test Qdrant in-memory mode"""
    print("\n" + "=" * 60)
    print("TEST 3: Qdrant Vector Database (In-Memory)")
    print("=" * 60)

    try:
        from modules.vector_db_module import VectorDatabase

        # Create in-memory database (se tu dong fallback)
        db = VectorDatabase(
            collection_name="test_collection",
            embedding_dimension=1536
        )

        # Test stats
        stats = db.get_collection_stats()
        print(f"Collection stats: {stats}")

        # Test add documents
        test_chunks = [
            {
                "text": "Day la cau thu nhat ve AI",
                "embedding": [0.1] * 1536,
                "chunk_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "audio_file": "test.mp3"
            },
            {
                "text": "Machine learning rat thu vi",
                "embedding": [0.2] * 1536,
                "chunk_id": 1,
                "start_time": 5.0,
                "end_time": 10.0,
                "audio_file": "test.mp3"
            }
        ]

        num_added = db.add_documents(test_chunks)
        print(f"\nAdded {num_added} documents")

        # Test search
        results = db.search(
            query_embedding=[0.15] * 1536,
            top_k=2
        )
        print(f"\nSearch results: {len(results)} documents")
        for r in results:
            print(f"  - {r['text']}: similarity={r['similarity']:.4f}")

        # Cleanup
        db.delete_collection()

        print("\n[PASS] Qdrant in-memory works correctly!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Qdrant error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_local():
    """Test Local Embedding module (Sentence-BERT/E5)"""
    print("\n" + "=" * 60)
    print("TEST 4a: Local Embedding Module (Sentence-BERT)")
    print("=" * 60)

    try:
        from modules.embedding_module import TextEmbedding

        print("Initializing LOCAL (Sentence-BERT) embedder...")
        embedder = TextEmbedding(provider="local", model_name="sbert")

        sample_texts = [
            "Tri tue nhan tao dang phat trien",
            "AI is developing rapidly",
            "Hom nay troi dep"
        ]

        print("Encoding sample texts...")
        embeddings = embedder.encode_text(sample_texts, show_progress=False)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedder.embedding_dim}")

        # Test similarity
        sim_12 = embedder.compute_similarity(embeddings[0], embeddings[1])
        sim_13 = embedder.compute_similarity(embeddings[0], embeddings[2])
        print(f"Similarity (AI-vn, AI-en): {sim_12:.4f}")
        print(f"Similarity (AI-vn, weather): {sim_13:.4f}")

        # Test encode_query
        query_emb = embedder.encode_query("AI la gi?")
        print(f"Query embedding length: {len(query_emb)}")

        print("\n[PASS] Local Embedding (Sentence-BERT) works correctly!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Local embedding error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_module():
    """Test Embedding module (ho tro ca OpenAI va Google)"""
    print("\n" + "=" * 60)
    print("TEST 4b: Cloud Embedding Module (OpenAI/Google)")
    print("=" * 60)

    # Check which provider to use
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Skip if using local provider and no cloud keys
    if provider == "local":
        if google_key:
            provider = "google"
        elif openai_key:
            provider = "openai"
        else:
            print("[SKIP] No cloud API keys - local embedding tested above")
            return None

    if provider == "google" and google_key:
        print(f"Using Google provider with key: {google_key[:10]}...")
    elif provider == "openai" and openai_key:
        print(f"Using OpenAI provider")
    elif google_key:
        provider = "google"
        print(f"Fallback to Google provider")
    elif openai_key:
        provider = "openai"
        print(f"Fallback to OpenAI provider")
    else:
        print("[SKIP] No API key found (GOOGLE_API_KEY or OPENAI_API_KEY)")
        return None

    try:
        from modules.embedding_module import TextEmbedding

        print(f"\nInitializing {provider.upper()} embedder...")
        embedder = TextEmbedding(provider=provider)

        # Test single text
        sample_texts = [
            "Tri tue nhan tao dang phat trien",
            "AI is developing rapidly",
            "Hom nay troi dep"
        ]

        print("Encoding sample texts...")
        embeddings = embedder.encode_text(sample_texts, show_progress=False)
        print(f"Embeddings shape: {embeddings.shape}")

        # Test similarity
        sim_12 = embedder.compute_similarity(embeddings[0], embeddings[1])
        sim_13 = embedder.compute_similarity(embeddings[0], embeddings[2])
        print(f"Similarity (text1, text2): {sim_12:.4f}")
        print(f"Similarity (text1, text3): {sim_13:.4f}")

        # Test encode_query
        query_emb = embedder.encode_query("AI la gi?")
        print(f"Query embedding length: {len(query_emb)}")

        print(f"\n[PASS] {provider.upper()} Embedding module works correctly!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Embedding error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline_mock():
    """Test full pipeline voi mock data (khong can API key)"""
    print("\n" + "=" * 60)
    print("TEST 5: Full Pipeline Integration (Mock)")
    print("=" * 60)

    try:
        from modules.vector_db_module import VectorDatabase
        from modules.chunking_module import TextChunker

        # 1. Chunking
        chunker = TextChunker(chunk_size=200, chunk_overlap=20, method="recursive")

        transcript = {
            "audio_filename": "lecture.mp3",
            "segments": [
                {"id": 0, "text": "Hom nay chung ta se hoc ve AI.", "start": 0.0, "end": 3.0},
                {"id": 1, "text": "AI la tri tue nhan tao.", "start": 3.0, "end": 6.0},
                {"id": 2, "text": "No dang thay doi the gioi.", "start": 6.0, "end": 9.0},
            ]
        }

        chunks = chunker.chunk_transcript(transcript)
        print(f"Created {len(chunks)} chunks")

        # 2. Mock embeddings
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = [float(i) * 0.1] * 1536

        # 3. Store in Qdrant
        db = VectorDatabase(
            collection_name="test_pipeline",
            embedding_dimension=1536
        )

        num_stored = db.add_documents(chunks)
        print(f"Stored {num_stored} chunks in Qdrant")

        # 4. Search
        query_emb = [0.05] * 1536
        results = db.search(query_emb, top_k=2)
        print(f"\nSearch results for mock query:")
        for r in results:
            print(f"  - {r['text'][:50]}... (sim={r['similarity']:.4f})")
            if r.get('metadata', {}).get('start_time'):
                print(f"    Time: {r['metadata']['start_time']} - {r['metadata']['end_time']}")

        # Cleanup
        db.delete_collection()

        print("\n[PASS] Full pipeline integration works correctly!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS")
    print("Qdrant + Local/Cloud Embeddings + LangChain")
    print("=" * 60)

    results = {}

    # Test 1: Config
    results["config"] = test_config()

    # Test 2: Chunking (basic)
    results["chunking_basic"] = test_chunking_basic()

    # Test 3: Qdrant in-memory
    results["qdrant_inmemory"] = test_vector_db_inmemory()

    # Test 4a: Local Embedding (Sentence-BERT/E5 - no API key needed)
    results["embedding_local"] = test_embedding_local()

    # Test 4b: Cloud Embedding (OpenAI/Google - needs API key)
    results["embedding_cloud"] = test_embedding_module()

    # Test 5: Full pipeline mock
    results["pipeline_mock"] = test_full_pipeline_mock()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"  {name}: {status}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
