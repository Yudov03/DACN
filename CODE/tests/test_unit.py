"""
Unit Tests - Test individual modules
=====================================

Tests for each module in isolation.
Run with: pytest tests/test_unit.py -v
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# 1. CHUNKING MODULE
# =============================================================================

class TestChunking:
    """Tests for TextChunker module"""

    def test_import(self):
        """Test import"""
        from src.modules import TextChunker
        assert TextChunker is not None

    def test_fixed_chunking(self):
        """Test fixed-size chunking"""
        from src.modules import TextChunker

        chunker = TextChunker(chunk_size=100, chunk_overlap=20, method="fixed")
        text = "This is a test sentence. " * 20
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("chunk_id" in c for c in chunks)

    def test_sentence_chunking(self):
        """Test sentence-based chunking"""
        from src.modules import TextChunker

        chunker = TextChunker(chunk_size=200, method="sentence")
        text = "First sentence. Second sentence. Third sentence. " * 10
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0

    def test_recursive_chunking(self):
        """Test recursive chunking"""
        from src.modules import TextChunker

        chunker = TextChunker(chunk_size=150, method="recursive")
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 5
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0

    def test_metadata_preservation(self):
        """Test metadata is preserved in chunks"""
        from src.modules import TextChunker

        chunker = TextChunker(chunk_size=100)
        text = "Test text for chunking."
        metadata = {"source": "test", "author": "unit_test"}
        chunks = chunker.chunk_text(text, metadata=metadata)

        assert len(chunks) > 0
        assert chunks[0].get("source") == "test"


# =============================================================================
# 2. EMBEDDING MODULE
# =============================================================================

class TestEmbedding:
    """Tests for TextEmbedding module"""

    def test_import(self):
        """Test import"""
        from src.modules import TextEmbedding
        assert TextEmbedding is not None

    def test_local_embedding_init(self):
        """Test local embedding initialization"""
        from src.modules import TextEmbedding

        embedder = TextEmbedding(provider="local", model_name="sbert")
        assert embedder.embedding_dim > 0

    def test_encode_single_text(self):
        """Test encoding single text"""
        from src.modules import TextEmbedding

        embedder = TextEmbedding(provider="local")
        embedding = embedder.encode_text("Test text")

        assert len(embedding) == embedder.embedding_dim

    def test_encode_multiple_texts(self):
        """Test encoding multiple texts"""
        from src.modules import TextEmbedding

        embedder = TextEmbedding(provider="local")
        texts = ["Text one", "Text two", "Text three"]
        embeddings = embedder.encode_text(texts)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == embedder.embedding_dim

    def test_encode_query(self):
        """Test encoding query"""
        from src.modules import TextEmbedding

        embedder = TextEmbedding(provider="local")
        query_emb = embedder.encode_query("What is AI?")

        assert len(query_emb) == embedder.embedding_dim

    def test_similarity(self):
        """Test similarity computation"""
        from src.modules import TextEmbedding

        embedder = TextEmbedding(provider="local")
        emb1 = embedder.encode_text("Machine learning")
        emb2 = embedder.encode_text("Artificial intelligence")
        emb3 = embedder.encode_text("Cooking recipes")

        sim_related = embedder.compute_similarity(emb1, emb2)
        sim_unrelated = embedder.compute_similarity(emb1, emb3)

        # Related texts should have higher similarity
        assert sim_related > sim_unrelated


# =============================================================================
# 3. VECTOR DATABASE MODULE
# =============================================================================

class TestVectorDB:
    """Tests for VectorDatabase module"""

    def test_import(self):
        """Test import"""
        from src.modules import VectorDatabase
        assert VectorDatabase is not None

    def test_init_inmemory(self):
        """Test in-memory initialization"""
        from src.modules import VectorDatabase

        db = VectorDatabase(
            collection_name="test_unit_db",
            embedding_dimension=768
        )
        assert db is not None
        db.delete_collection()

    def test_add_and_search(self):
        """Test adding documents and searching"""
        from src.modules import VectorDatabase

        db = VectorDatabase(
            collection_name="test_unit_search",
            embedding_dimension=768
        )

        # Add documents
        chunks = [
            {"id": "1", "text": "Machine learning", "embedding": [0.1] * 768},
            {"id": "2", "text": "Deep learning", "embedding": [0.2] * 768},
            {"id": "3", "text": "Neural networks", "embedding": [0.15] * 768},
        ]
        db.add_documents(chunks)

        # Search
        results = db.search(query_embedding=[0.12] * 768, top_k=2)

        assert len(results) == 2
        assert all("text" in r for r in results)
        assert all("similarity" in r for r in results)

        db.delete_collection()

    def test_collection_stats(self):
        """Test getting collection stats"""
        from src.modules import VectorDatabase

        db = VectorDatabase(
            collection_name="test_unit_stats",
            embedding_dimension=768
        )

        chunks = [{"id": "1", "text": "Test", "embedding": [0.1] * 768}]
        db.add_documents(chunks)

        stats = db.get_collection_stats()
        assert stats.get("count", 0) >= 1

        db.delete_collection()


# =============================================================================
# 4. DOCUMENT PROCESSOR MODULE
# =============================================================================

class TestDocumentProcessor:
    """Tests for Document Processor module"""

    def test_import(self):
        """Test imports"""
        from src.modules import UnifiedProcessor, TextProcessor
        assert UnifiedProcessor is not None
        assert TextProcessor is not None

    def test_supported_formats(self):
        """Test supported formats count"""
        from src.modules import UnifiedProcessor

        processor = UnifiedProcessor()
        extensions = processor.supported_extensions()

        assert len(extensions) >= 30  # Should support 34 formats

    def test_text_processor(self):
        """Test TextProcessor"""
        from src.modules import TextProcessor

        processor = TextProcessor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("This is a test document.\n\nIt has multiple paragraphs.")
            temp_path = f.name

        try:
            result = processor.process(temp_path)
            assert result.content is not None
            assert len(result.content) > 0
        finally:
            Path(temp_path).unlink()

    def test_unified_processor_text(self):
        """Test UnifiedProcessor with text file"""
        from src.modules import UnifiedProcessor

        processor = UnifiedProcessor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Test document content for unified processor.")
            temp_path = f.name

        try:
            result = processor.process(temp_path)
            assert result is not None
            assert result.content is not None
        finally:
            Path(temp_path).unlink()


# =============================================================================
# 5. KNOWLEDGE BASE MODULE
# =============================================================================

class TestKnowledgeBase:
    """Tests for KnowledgeBase module"""

    def test_import(self):
        """Test import"""
        from src.modules import KnowledgeBase
        assert KnowledgeBase is not None

    def test_init(self):
        """Test initialization"""
        from src.modules import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(base_dir=tmpdir)
            assert kb is not None

    def test_stats(self):
        """Test getting stats"""
        from src.modules import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(base_dir=tmpdir)
            stats = kb.get_stats()

            assert hasattr(stats, 'total_documents')
            assert hasattr(stats, 'total_chunks')
            assert stats.total_documents >= 0

    def test_add_document(self):
        """Test adding document"""
        from src.modules import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(base_dir=tmpdir)

            # Create test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test document content", encoding='utf-8')

            doc_id = kb.add_document(str(test_file))
            assert doc_id is not None

            stats = kb.get_stats()
            assert stats.total_documents >= 1


# =============================================================================
# 6. TTS MODULE
# =============================================================================

class TestTTS:
    """Tests for Text-to-Speech module"""

    def test_import(self):
        """Test import"""
        from src.modules import TextToSpeech
        assert TextToSpeech is not None

    def test_init(self):
        """Test initialization"""
        from src.modules import TextToSpeech

        tts = TextToSpeech(voice="vi-female")
        assert tts.voice == "vi-VN-HoaiMyNeural"

    def test_voice_shortcuts(self):
        """Test voice shortcuts"""
        from src.modules import TextToSpeech

        shortcuts = TextToSpeech.get_available_shortcuts()

        assert "vi-female" in shortcuts
        assert "vi-male" in shortcuts
        assert "en-female" in shortcuts
        assert "en-male" in shortcuts

    def test_list_voices(self):
        """Test listing voices"""
        from src.modules import TextToSpeech

        voices = TextToSpeech.list_voices_sync()
        assert len(voices) > 0

    def test_synthesize(self):
        """Test speech synthesis"""
        from src.modules import TextToSpeech

        tts = TextToSpeech(voice="vi-female")
        audio = tts.synthesize_sync("Xin chao")

        assert audio is not None
        assert len(audio) > 0
        assert isinstance(audio, bytes)

    def test_empty_text(self):
        """Test empty text handling"""
        from src.modules import TextToSpeech

        tts = TextToSpeech()
        audio = tts.synthesize_sync("")

        assert audio == b""


# =============================================================================
# 7. ANSWER VERIFICATION MODULE (Anti-Hallucination)
# =============================================================================

class TestAnswerVerification:
    """Tests for Answer Verification module"""

    def test_import(self):
        """Test imports"""
        from src.modules import AnswerVerifier, AbstentionChecker
        assert AnswerVerifier is not None
        assert AbstentionChecker is not None

    def test_verifier_init(self):
        """Test verifier initialization"""
        from src.modules import AnswerVerifier

        verifier = AnswerVerifier()
        assert verifier is not None

    def test_verify_grounded_answer(self):
        """Test verifying a grounded answer"""
        from src.modules import AnswerVerifier

        verifier = AnswerVerifier()
        context = "The minimum passing score is 5.0 points."
        answer = "The minimum score is 5.0."
        question = "What is the minimum score?"

        result = verifier.verify(answer, context, question)

        assert result.confidence_score >= 0
        assert result.confidence_score <= 1

    def test_abstention_checker(self):
        """Test abstention checker"""
        from src.modules import AbstentionChecker

        checker = AbstentionChecker(min_retrieval_score=0.5)

        # Low similarity should trigger abstention
        should_abstain, reason = checker.should_abstain(
            "Question?",
            [{"similarity": 0.2}]
        )

        assert should_abstain == True
        assert reason is not None


# =============================================================================
# 8. CONFLICT DETECTION MODULE
# =============================================================================

class TestConflictDetection:
    """Tests for Conflict Detection module"""

    def test_import(self):
        """Test import"""
        from src.modules import ConflictDetector
        assert ConflictDetector is not None

    def test_detector_init(self):
        """Test detector initialization"""
        from src.modules import ConflictDetector

        detector = ConflictDetector()
        assert detector is not None

    def test_detect_conflicts(self):
        """Test conflict detection"""
        from src.modules import ConflictDetector

        detector = ConflictDetector()

        chunks = [
            {"text": "Fee 2023: 15 million", "metadata": {"date": "2023-01-01"}, "similarity": 0.8},
            {"text": "Fee 2024: 18 million", "metadata": {"date": "2024-01-01"}, "similarity": 0.85},
        ]

        result = detector.detect_and_resolve(chunks, "fee")

        assert hasattr(result, 'has_conflicts')
        assert hasattr(result, 'resolved_context')

    def test_date_extraction(self):
        """Test date extraction from text"""
        from src.modules import ConflictDetector

        detector = ConflictDetector()
        date = detector._extract_date("Policy dated 2024-06-15", {})

        assert date is not None
        assert date.year == 2024


# =============================================================================
# 9. PROMPT TEMPLATES MODULE
# =============================================================================

class TestPromptTemplates:
    """Tests for Prompt Templates module"""

    def test_import(self):
        """Test import"""
        from src.modules import PromptTemplateManager
        assert PromptTemplateManager is not None

    def test_init(self):
        """Test initialization"""
        from src.modules import PromptTemplateManager

        manager = PromptTemplateManager(language="vi")
        assert manager is not None

    def test_list_templates(self):
        """Test listing templates"""
        from src.modules import PromptTemplateManager

        manager = PromptTemplateManager()
        templates = manager.list_templates()

        assert len(templates) >= 9  # Should have 9 templates
        assert "strict_qa" in templates
        assert "citation_required" in templates
        assert "safe_abstention" in templates

    def test_format_prompt(self):
        """Test formatting prompt"""
        from src.modules import PromptTemplateManager

        manager = PromptTemplateManager()
        sys_prompt, user_prompt = manager.format_prompt(
            "strict_qa",
            context="Test context",
            question="Test question?"
        )

        assert len(sys_prompt) > 0
        assert len(user_prompt) > 0
        assert "Test context" in user_prompt or "Test question" in user_prompt


# =============================================================================
# 10. RAG MODULE
# =============================================================================

class TestRAG:
    """Tests for RAG module"""

    def test_import(self):
        """Test import"""
        from src.modules import RAGSystem
        assert RAGSystem is not None

    def test_enhanced_features_flag(self):
        """Test enhanced features flag"""
        from src.modules.rag_module import HAS_ENHANCED_FEATURES
        assert HAS_ENHANCED_FEATURES == True


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_unit_tests():
    """Run all unit tests manually"""
    import traceback

    test_classes = [
        TestChunking,
        TestEmbedding,
        TestVectorDB,
        TestDocumentProcessor,
        TestKnowledgeBase,
        TestTTS,
        TestAnswerVerification,
        TestConflictDetection,
        TestPromptTemplates,
        TestRAG,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    print("=" * 70)
    print("UNIT TESTS")
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
    success = run_all_unit_tests()
    sys.exit(0 if success else 1)
