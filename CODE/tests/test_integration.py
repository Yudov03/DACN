"""
Integration Tests - Test module interactions
=============================================

Tests for pipelines and module combinations.
Run with: pytest tests/test_integration.py -v
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# 1. CHUNKING + EMBEDDING PIPELINE
# =============================================================================

class TestChunkingEmbeddingPipeline:
    """Tests for Chunking -> Embedding pipeline"""

    def test_chunk_and_embed(self):
        """Test chunking text then embedding chunks"""
        from src.modules import TextChunker, TextEmbedding

        # Chunk
        chunker = TextChunker(chunk_size=100)
        text = "Artificial intelligence is transforming the world. " * 10
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0

        # Embed
        embedder = TextEmbedding(provider="local")
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_text(texts)

        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == embedder.embedding_dim

    def test_chunk_embed_with_metadata(self):
        """Test chunking with metadata preserved through embedding"""
        from src.modules import TextChunker, TextEmbedding

        chunker = TextChunker(chunk_size=100)
        text = "Test content for metadata preservation."
        metadata = {"source": "test_file.txt", "author": "test"}
        chunks = chunker.chunk_text(text, metadata=metadata)

        embedder = TextEmbedding(provider="local")

        for chunk in chunks:
            chunk["embedding"] = embedder.encode_text(chunk["text"]).tolist()

        assert all("embedding" in c for c in chunks)
        assert all(c.get("source") == "test_file.txt" for c in chunks)


# =============================================================================
# 2. EMBEDDING + VECTOR DB PIPELINE
# =============================================================================

class TestEmbeddingVectorDBPipeline:
    """Tests for Embedding -> VectorDB pipeline"""

    def test_embed_and_store(self):
        """Test embedding texts and storing in vector DB"""
        from src.modules import TextEmbedding, VectorDatabase

        embedder = TextEmbedding(provider="local")
        db = VectorDatabase(
            collection_name="test_integration_embed_store",
            embedding_dimension=embedder.embedding_dim
        )

        try:
            # Create and embed chunks
            texts = ["Machine learning basics", "Deep learning networks", "AI applications"]
            embeddings = embedder.encode_text(texts)

            chunks = []
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                chunks.append({
                    "id": str(i),
                    "text": text,
                    "embedding": emb.tolist()
                })

            # Store
            db.add_documents(chunks)

            # Verify
            stats = db.get_collection_stats()
            assert stats.get("count", 0) == 3

        finally:
            db.delete_collection()

    def test_semantic_search(self):
        """Test semantic search with embeddings"""
        from src.modules import TextEmbedding, VectorDatabase

        embedder = TextEmbedding(provider="local")
        db = VectorDatabase(
            collection_name="test_integration_semantic",
            embedding_dimension=embedder.embedding_dim
        )

        try:
            # Add documents
            docs = [
                "Python is a programming language",
                "Machine learning uses algorithms",
                "Cooking requires ingredients",
                "Deep learning is a subset of ML",
            ]
            embeddings = embedder.encode_text(docs)

            chunks = [
                {"id": str(i), "text": t, "embedding": e.tolist()}
                for i, (t, e) in enumerate(zip(docs, embeddings))
            ]
            db.add_documents(chunks)

            # Search for ML-related content
            query = "artificial intelligence and neural networks"
            query_emb = embedder.encode_query(query)
            results = db.search(query_embedding=query_emb, top_k=2)

            # ML-related docs should rank higher than cooking
            assert len(results) == 2
            result_texts = [r["text"] for r in results]
            assert "Cooking" not in " ".join(result_texts)

        finally:
            db.delete_collection()


# =============================================================================
# 3. FULL RETRIEVAL PIPELINE
# =============================================================================

class TestFullRetrievalPipeline:
    """Tests for complete retrieval pipeline"""

    def test_chunk_embed_store_search(self):
        """Test full pipeline: Chunk -> Embed -> Store -> Search"""
        from src.modules import TextChunker, TextEmbedding, VectorDatabase

        chunker = TextChunker(chunk_size=200)
        embedder = TextEmbedding(provider="local")
        db = VectorDatabase(
            collection_name="test_full_pipeline",
            embedding_dimension=embedder.embedding_dim
        )

        try:
            # 1. Chunk document
            document = """
            Introduction to Machine Learning.
            Machine learning is a subset of artificial intelligence.
            It enables computers to learn from data.

            Types of Machine Learning.
            Supervised learning uses labeled data.
            Unsupervised learning finds patterns in unlabeled data.
            Reinforcement learning learns through rewards.
            """

            chunks = chunker.chunk_text(document.strip(), metadata={"source": "ml_guide.txt"})
            assert len(chunks) > 0

            # 2. Embed chunks
            for chunk in chunks:
                emb = embedder.encode_text(chunk["text"])
                chunk["embedding"] = emb.tolist()

            # 3. Store in vector DB
            db.add_documents(chunks)

            # 4. Search
            query = "What are the types of machine learning?"
            query_emb = embedder.encode_query(query)
            results = db.search(query_embedding=query_emb, top_k=2)

            assert len(results) > 0
            # Should find relevant content about ML types
            assert any("learning" in r["text"].lower() for r in results)

        finally:
            db.delete_collection()


# =============================================================================
# 4. DOCUMENT PROCESSOR + KNOWLEDGE BASE PIPELINE
# =============================================================================

class TestDocProcessorKBPipeline:
    """Tests for Document Processor -> Knowledge Base pipeline"""

    def test_process_and_add_to_kb(self):
        """Test processing document and adding to KB"""
        from src.modules import UnifiedProcessor, KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test document
            doc_path = Path(tmpdir) / "test_doc.txt"
            doc_path.write_text(
                "This is a test document.\n\n"
                "It contains important information.\n\n"
                "Used for testing the pipeline.",
                encoding="utf-8"
            )

            # Process with UnifiedProcessor
            processor = UnifiedProcessor()
            result = processor.process(str(doc_path))

            assert result is not None
            assert result.content is not None

            # Add to Knowledge Base
            kb_dir = Path(tmpdir) / "kb"
            kb_dir.mkdir()
            kb = KnowledgeBase(base_dir=str(kb_dir))

            doc_id = kb.add_document(str(doc_path))
            assert doc_id is not None

            # Verify
            stats = kb.get_stats()
            assert stats.total_documents >= 1


# =============================================================================
# 5. ANTI-HALLUCINATION PIPELINE
# =============================================================================

class TestAntiHallucinationPipeline:
    """Tests for Answer Verification + Conflict Detection pipeline"""

    def test_verify_and_check_abstention(self):
        """Test verification with abstention check"""
        from src.modules import AnswerVerifier, AbstentionChecker

        verifier = AnswerVerifier()
        checker = AbstentionChecker(min_retrieval_score=0.5)

        # Scenario 1: Good context, should answer
        good_contexts = [{"similarity": 0.8, "text": "The answer is 42."}]
        should_abstain, _ = checker.should_abstain("What is the answer?", good_contexts)
        assert should_abstain == False

        # Scenario 2: Poor context, should abstain
        poor_contexts = [{"similarity": 0.2, "text": "Unrelated content."}]
        should_abstain, reason = checker.should_abstain("Complex question?", poor_contexts)
        assert should_abstain == True

    def test_conflict_then_verify(self):
        """Test conflict detection followed by verification"""
        from src.modules import ConflictDetector, AnswerVerifier

        detector = ConflictDetector()
        verifier = AnswerVerifier()

        # Detect conflicts
        chunks = [
            {"text": "Price: $100", "metadata": {"date": "2023-01"}, "similarity": 0.8},
            {"text": "Price: $120 (updated)", "metadata": {"date": "2024-01"}, "similarity": 0.85},
        ]

        conflict_result = detector.detect_and_resolve(chunks, "price")

        # Use resolved context for answer
        if conflict_result.resolved_context:
            context = conflict_result.resolved_context
        else:
            context = " ".join([c["text"] for c in chunks])

        answer = "The price is $120."
        verify_result = verifier.verify(answer, context, "What is the price?")
        assert verify_result.confidence_score >= 0


# =============================================================================
# 6. TTS INTEGRATION
# =============================================================================

class TestTTSIntegration:
    """Tests for TTS integration with other modules"""

    def test_text_to_speech_flow(self):
        """Test generating speech from processed text"""
        from src.modules import TextToSpeech

        tts = TextToSpeech(voice="vi-female")

        # Simulate answer from RAG
        answer = "Day la cau tra loi tu he thong."

        audio = tts.synthesize_sync(answer)

        assert audio is not None
        assert len(audio) > 0
        assert isinstance(audio, bytes)

    def test_tts_settings(self):
        """Test TTS with different settings"""
        from src.modules import TextToSpeech

        tts = TextToSpeech(voice="en-female", rate="+20%")

        tts.set_voice("vi-male")
        assert tts.voice == "vi-VN-NamMinhNeural"

        tts.set_rate("-10%")
        assert tts.rate == "-10%"


# =============================================================================
# 7. PROMPT TEMPLATES INTEGRATION
# =============================================================================

class TestPromptIntegration:
    """Tests for Prompt Templates integration"""

    def test_prompt_with_context(self):
        """Test formatting prompts with real context"""
        from src.modules import PromptTemplateManager

        manager = PromptTemplateManager(language="vi")

        context = """
        Quy dinh ve hoc phi nam 2024:
        - Hoc phi co ban: 15 trieu dong/ky
        - Hoc phi tin chi: 500,000 dong/tin chi
        """

        question = "Hoc phi mot ky la bao nhieu?"

        sys_prompt, user_prompt = manager.format_prompt(
            "strict_qa",
            context=context,
            question=question
        )

        assert "15 trieu" in user_prompt or "Hoc phi" in user_prompt
        assert len(sys_prompt) > 0

    def test_multiple_templates(self):
        """Test different templates produce different prompts"""
        from src.modules import PromptTemplateManager

        manager = PromptTemplateManager()

        templates = ["basic_qa", "strict_qa", "citation_required"]
        prompts = []

        for template in templates:
            sys_p, user_p = manager.format_prompt(
                template,
                context="Test context",
                question="Test question?"
            )
            prompts.append(sys_p)

        # Different templates should produce different system prompts
        assert len(set(prompts)) == len(templates)


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_integration_tests():
    """Run all integration tests manually"""

    test_classes = [
        TestChunkingEmbeddingPipeline,
        TestEmbeddingVectorDBPipeline,
        TestFullRetrievalPipeline,
        TestDocProcessorKBPipeline,
        TestAntiHallucinationPipeline,
        TestTTSIntegration,
        TestPromptIntegration,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    print("=" * 70)
    print("INTEGRATION TESTS")
    print("=" * 70)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  [PASS] {method_name}")
                total_passed += 1
            except Exception as e:
                print(f"  [FAIL] {method_name}: {e}")
                total_failed += 1
                failures.append((test_class.__name__, method_name, str(e)))

    print("\n" + "=" * 70)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    if failures:
        print("\nFailures:")
        for cls, method, error in failures:
            print(f"  - {cls}.{method}: {error}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
