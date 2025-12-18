# Tests Directory

Test suite for the Audio Information Retrieval system.

## Directory Structure

```
tests/
├── conftest.py         # Pytest fixtures
├── run_tests.py        # Test runner script
├── test_unit.py        # Unit tests (43 tests)
├── test_integration.py # Integration tests (12 tests)
├── test_e2e.py         # End-to-end tests (9 tests)
└── test_data/          # Test data files
```

## Test Organization

### Test Pyramid

```
        ┌─────────┐
        │   E2E   │  9 tests  - Full workflows
        │  Tests  │
       ─┴─────────┴─
      ┌─────────────┐
      │ Integration │  12 tests - Module pipelines
      │    Tests    │
     ─┴─────────────┴─
    ┌─────────────────┐
    │   Unit Tests    │  43 tests - Individual modules
    └─────────────────┘
```

**Total: 64 tests**

## Test Files

### test_unit.py (43 tests)
Unit tests for individual modules in isolation.

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestChunking` | 5 | TextChunker (fixed, sentence, recursive) |
| `TestEmbedding` | 6 | TextEmbedding (local, encode, similarity) |
| `TestVectorDB` | 4 | VectorDatabase (init, add, search, stats) |
| `TestDocumentProcessor` | 4 | UnifiedProcessor, TextProcessor |
| `TestKnowledgeBase` | 4 | KnowledgeBase (init, add, stats) |
| `TestTTS` | 6 | TextToSpeech (voices, synthesis) |
| `TestAnswerVerification` | 4 | AnswerVerifier, AbstentionChecker |
| `TestConflictDetection` | 4 | ConflictDetector |
| `TestPromptTemplates` | 4 | PromptTemplateManager (9 templates) |
| `TestRAG` | 2 | RAGSystem, enhanced features |

### test_integration.py (12 tests)
Integration tests for module pipelines.

| Class | Tests | Pipeline |
|-------|-------|----------|
| `TestChunkingEmbeddingPipeline` | 2 | Chunking → Embedding |
| `TestEmbeddingVectorDBPipeline` | 2 | Embedding → VectorDB |
| `TestFullRetrievalPipeline` | 1 | Chunk → Embed → Store → Search |
| `TestDocProcessorKBPipeline` | 1 | Document → Knowledge Base |
| `TestAntiHallucinationPipeline` | 2 | Verification + Conflict Detection |
| `TestTTSIntegration` | 2 | TTS with settings |
| `TestPromptIntegration` | 2 | Prompt templates with context |

### test_e2e.py (9 tests)
End-to-end tests for complete workflows.

| Class | Tests | Workflow |
|-------|-------|----------|
| `TestDocumentToAnswerFlow` | 1 | Document → Process → Store → Query → Answer |
| `TestKnowledgeBaseWorkflow` | 1 | KB add → search → export → import |
| `TestAntiHallucinationWorkflow` | 2 | Hallucination detection, conflict resolution |
| `TestTTSOutputWorkflow` | 2 | Answer → Speech conversion |
| `TestMultiFormatWorkflow` | 1 | Process multiple file formats |
| `TestSystemHealthCheck` | 2 | All imports, enhanced features |

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests (fast)
python tests/run_tests.py quick

# Using pytest
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Unit tests only
pytest tests/test_unit.py -v

# Integration tests only
pytest tests/test_integration.py -v

# E2E tests only
pytest tests/test_e2e.py -v
```

### Run Specific Test Class

```bash
# Test specific module
pytest tests/test_unit.py::TestChunking -v
pytest tests/test_unit.py::TestTTS -v
pytest tests/test_unit.py::TestAnswerVerification -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Fixtures (conftest.py)

Shared fixtures available for all tests:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_data_dir` | session | Path to test_data/ folder |
| `temp_dir` | function | Temporary directory (auto-cleanup) |
| `project_root` | session | Project root path |
| `sample_texts` | session | Vietnamese sample texts |
| `sample_english_texts` | session | English sample texts |
| `sample_text_file` | function | Creates temp text file |
| `embedder` | session | Shared TextEmbedding instance |
| `knowledge_base` | function | Temp KnowledgeBase instance |
| `tts` | function | TextToSpeech instance |
| `document_processor` | function | UnifiedProcessor instance |
| `vector_db` | function | Temp VectorDatabase (auto-cleanup) |

## Test Data

```
test_data/
└── (test files created dynamically)
```

Most tests create temporary files/directories that are automatically cleaned up.

## Writing New Tests

### Unit Test Example

```python
# tests/test_unit.py

class TestNewModule:
    """Tests for NewModule"""

    def test_import(self):
        """Test import works"""
        from src.modules import NewModule
        assert NewModule is not None

    def test_basic_functionality(self):
        """Test basic functionality"""
        from src.modules import NewModule

        module = NewModule()
        result = module.process("input")

        assert result is not None
        assert len(result) > 0
```

### Integration Test Example

```python
# tests/test_integration.py

class TestNewPipeline:
    """Tests for Module A → Module B pipeline"""

    def test_pipeline_flow(self):
        """Test complete pipeline"""
        from src.modules import ModuleA, ModuleB

        # Step 1
        a = ModuleA()
        intermediate = a.process("input")

        # Step 2
        b = ModuleB()
        result = b.process(intermediate)

        assert result is not None
```

## Notes

- Tests use **local embedding** (SBERT) to avoid API dependencies
- Vector database tests **auto-cleanup** collections after each test
- TTS tests use **edge-tts** (no API key required)
- Some tests may take longer on first run (model downloads)
- Tests read config from `.env` file automatically
- Qdrant must be running for vector database tests (`docker run -p 6333:6333 qdrant/qdrant`)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Qdrant connection refused` | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` |
| `UnicodeEncodeError` | Use `python tests/run_tests.py` instead of `pytest` directly |
| `First run slow` | Models are downloading - this is normal |
