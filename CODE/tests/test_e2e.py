"""
End-to-End Tests - Full system tests
=====================================

Tests for complete workflows with real data.
Run with: pytest tests/test_e2e.py -v
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# 1. DOCUMENT TO ANSWER FLOW
# =============================================================================

class TestDocumentToAnswerFlow:
    """End-to-end test: Document -> Process -> Store -> Query -> Answer"""

    def test_text_document_qa(self):
        """Test full Q&A flow with text document"""
        from src.modules import (
            UnifiedProcessor, TextChunker, TextEmbedding,
            VectorDatabase, PromptTemplateManager
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create document
            doc_path = Path(tmpdir) / "regulations.txt"
            doc_path.write_text("""
            Student Regulations 2024

            Article 1: Tuition Fees
            The tuition fee for the academic year 2024 is 15 million VND per semester.
            Students can pay in two installments.

            Article 2: Attendance
            Students must attend at least 80% of classes.
            Absence without valid reason will result in course failure.

            Article 3: Examinations
            Final exams account for 60% of the total grade.
            Mid-term exams account for 40% of the total grade.
            """, encoding="utf-8")

            # 2. Process document
            processor = UnifiedProcessor()
            result = processor.process(str(doc_path))
            assert result.content is not None

            # 3. Chunk
            chunker = TextChunker(chunk_size=200)
            chunks = chunker.chunk_text(result.content, metadata={"source": "regulations.txt"})
            assert len(chunks) > 0

            # 4. Embed
            embedder = TextEmbedding(provider="local")
            for chunk in chunks:
                chunk["embedding"] = embedder.encode_text(chunk["text"]).tolist()

            # 5. Store
            db = VectorDatabase(
                collection_name="test_e2e_qa",
                embedding_dimension=embedder.embedding_dim
            )

            try:
                db.add_documents(chunks)

                # 6. Query
                question = "What is the tuition fee?"
                query_emb = embedder.encode_query(question)
                results = db.search(query_embedding=query_emb, top_k=3)

                assert len(results) > 0

                # 7. Prepare context and generate prompt
                context = "\n".join([r["text"] for r in results])
                manager = PromptTemplateManager()
                sys_prompt, user_prompt = manager.format_prompt(
                    "strict_qa", context=context, question=question
                )

                # Verify context contains relevant info
                assert "15 million" in context or "tuition" in context.lower()
                assert len(sys_prompt) > 0
                assert len(user_prompt) > 0

            finally:
                db.delete_collection()


# =============================================================================
# 2. KNOWLEDGE BASE WORKFLOW
# =============================================================================

class TestKnowledgeBaseWorkflow:
    """End-to-end test: Knowledge Base management workflow"""

    def test_kb_add_search_export_import(self):
        """Test full KB workflow: add -> search -> export -> import"""
        from src.modules import KnowledgeBase

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "kb"
            kb_dir.mkdir()

            # 1. Create KB
            kb = KnowledgeBase(base_dir=str(kb_dir))

            # 2. Create and add documents
            doc1 = Path(tmpdir) / "doc1.txt"
            doc1.write_text("Machine learning is a subset of AI.", encoding="utf-8")

            doc2 = Path(tmpdir) / "doc2.txt"
            doc2.write_text("Deep learning uses neural networks.", encoding="utf-8")

            doc_id1 = kb.add_document(str(doc1), tags=["ml", "ai"])
            doc_id2 = kb.add_document(str(doc2), tags=["dl", "ai"])

            assert doc_id1 is not None
            assert doc_id2 is not None

            # 3. Verify stats
            stats = kb.get_stats()
            assert stats.total_documents == 2

            # 4. Search (by tag, not content)
            results = kb.search_documents("ml")
            assert len(results) >= 1

            # 5. Export
            export_path = Path(tmpdir) / "backup.zip"
            kb.export_kb(str(export_path))
            assert export_path.exists()

            # 6. Create new KB and import
            kb2_dir = Path(tmpdir) / "kb2"
            kb2_dir.mkdir()
            kb2 = KnowledgeBase(base_dir=str(kb2_dir))

            kb2.import_kb(str(export_path))

            stats2 = kb2.get_stats()
            assert stats2.total_documents == 2


# =============================================================================
# 3. ANTI-HALLUCINATION WORKFLOW
# =============================================================================

class TestAntiHallucinationWorkflow:
    """End-to-end test: Anti-hallucination checks in RAG flow"""

    def test_hallucination_detection(self):
        """Test detecting potential hallucination"""
        from src.modules import (
            AnswerVerifier, AbstentionChecker, ConflictDetector
        )

        # Scenario: User asks about something not in context
        context = "The university was founded in 1956. It has 5 faculties."
        question = "What is the tuition fee?"
        answer = "The tuition fee is $5000 per year."  # Hallucinated!

        # 1. Check if we should abstain (simulated low retrieval score)
        checker = AbstentionChecker(min_retrieval_score=0.5)
        contexts = [{"similarity": 0.3, "text": context}]  # Low similarity
        should_abstain, reason = checker.should_abstain(question, contexts)

        # Should recommend abstention due to low relevance
        assert should_abstain == True

        # 2. Even if we proceed, verification should flag the answer
        verifier = AnswerVerifier()
        result = verifier.verify(answer, context, question)

        # Should have low confidence (tuition not in context)
        assert result.confidence_score < 0.8

    def test_conflict_resolution(self):
        """Test resolving conflicting information"""
        from src.modules import ConflictDetector

        detector = ConflictDetector()

        # Two documents with different dates
        chunks = [
            {
                "text": "Tuition 2023: 12 million VND",
                "metadata": {"date": "2023-01-01", "source": "old_doc.pdf"},
                "similarity": 0.85
            },
            {
                "text": "Tuition 2024: 15 million VND (updated regulation)",
                "metadata": {"date": "2024-01-01", "source": "new_doc.pdf"},
                "similarity": 0.82
            }
        ]

        result = detector.detect_and_resolve(chunks, "tuition fee")

        # Should have resolved context
        assert hasattr(result, 'resolved_context')
        assert hasattr(result, 'has_conflicts')

        # Check resolved context prefers newer info
        if result.resolved_context:
            assert "2024" in result.resolved_context or "15 million" in result.resolved_context


# =============================================================================
# 4. TTS OUTPUT WORKFLOW
# =============================================================================

class TestTTSOutputWorkflow:
    """End-to-end test: Generate answer and convert to speech"""

    def test_answer_to_speech(self):
        """Test converting answer text to speech"""
        from src.modules import TextToSpeech, PromptTemplateManager

        # 1. Simulate RAG answer
        answer = "Hoc phi nam 2024 la 15 trieu dong moi ky."

        # 2. Convert to speech
        tts = TextToSpeech(voice="vi-female")
        audio = tts.synthesize_sync(answer)

        assert audio is not None
        assert len(audio) > 1000  # Should have substantial audio data
        assert isinstance(audio, bytes)

    def test_multilingual_tts(self):
        """Test TTS with both Vietnamese and English"""
        from src.modules import TextToSpeech

        # Vietnamese
        tts_vi = TextToSpeech(voice="vi-female")
        audio_vi = tts_vi.synthesize_sync("Xin chao")
        assert len(audio_vi) > 0

        # English
        tts_en = TextToSpeech(voice="en-female")
        audio_en = tts_en.synthesize_sync("Hello")
        assert len(audio_en) > 0


# =============================================================================
# 5. MULTI-FORMAT DOCUMENT WORKFLOW
# =============================================================================

class TestMultiFormatWorkflow:
    """End-to-end test: Processing multiple document formats"""

    def test_process_multiple_formats(self):
        """Test processing different file formats"""
        from src.modules import UnifiedProcessor

        processor = UnifiedProcessor()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Text file
            txt_file = Path(tmpdir) / "test.txt"
            txt_file.write_text("Text document content.", encoding="utf-8")

            # Markdown file
            md_file = Path(tmpdir) / "test.md"
            md_file.write_text("# Markdown\n\nContent here.", encoding="utf-8")

            # JSON file
            json_file = Path(tmpdir) / "test.json"
            json_file.write_text('{"key": "value", "data": "test"}', encoding="utf-8")

            # Process each
            for file_path in [txt_file, md_file, json_file]:
                result = processor.process(str(file_path))
                assert result is not None
                assert result.content is not None
                assert len(result.content) > 0


# =============================================================================
# 6. FULL SYSTEM HEALTH CHECK
# =============================================================================

class TestSystemHealthCheck:
    """End-to-end test: Verify all components are working"""

    def test_all_imports(self):
        """Test all major imports work"""
        from src.modules import (
            # Core
            TextChunker,
            TextEmbedding,
            VectorDatabase,
            RAGSystem,
            # Document Processing
            UnifiedProcessor,
            TextProcessor,
            # Knowledge Base
            KnowledgeBase,
            # TTS
            TextToSpeech,
            # Anti-Hallucination
            AnswerVerifier,
            AbstentionChecker,
            ConflictDetector,
            # Optimization
            PromptTemplateManager,
        )

        assert all([
            TextChunker, TextEmbedding, VectorDatabase, RAGSystem,
            UnifiedProcessor, TextProcessor,
            KnowledgeBase,
            TextToSpeech,
            AnswerVerifier, AbstentionChecker, ConflictDetector,
            PromptTemplateManager,
        ])

    def test_enhanced_rag_features(self):
        """Test enhanced RAG features are available"""
        from src.modules.rag_module import HAS_ENHANCED_FEATURES
        assert HAS_ENHANCED_FEATURES == True

    def test_supported_formats_count(self):
        """Test correct number of supported formats"""
        from src.modules import UnifiedProcessor

        processor = UnifiedProcessor()
        extensions = processor.supported_extensions()

        assert len(extensions) >= 30  # Should support 34+ formats


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_e2e_tests():
    """Run all end-to-end tests manually"""

    test_classes = [
        TestDocumentToAnswerFlow,
        TestKnowledgeBaseWorkflow,
        TestAntiHallucinationWorkflow,
        TestTTSOutputWorkflow,
        TestMultiFormatWorkflow,
        TestSystemHealthCheck,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    print("=" * 70)
    print("END-TO-END TESTS")
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
    success = run_all_e2e_tests()
    sys.exit(0 if success else 1)
