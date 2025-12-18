"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for all tests.
"""

import pytest
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Return test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir():
    """Create temporary directory for each test"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory"""
    return Path(__file__).parent.parent


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_texts():
    """Sample Vietnamese texts for testing"""
    return [
        "Quy dinh ve dang ky mon hoc trong hoc ky moi",
        "Cach tinh diem trung binh tich luy cua sinh vien",
        "Thoi han nop hoc phi va cac khoan phi khac",
        "Quy che thi cu va kiem tra giua ky",
        "Huong dan lam do an tot nghiep cho sinh vien nam cuoi",
    ]


@pytest.fixture(scope="session")
def sample_english_texts():
    """Sample English texts for testing"""
    return [
        "Guidelines for course registration",
        "How to calculate cumulative GPA",
        "Tuition payment deadlines",
        "Examination rules and regulations",
        "Thesis writing guidelines",
    ]


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file"""
    file_path = temp_dir / "sample.txt"
    file_path.write_text(
        "This is a sample document.\n\n"
        "It contains multiple paragraphs.\n\n"
        "Used for testing the document processing pipeline.",
        encoding="utf-8"
    )
    return file_path


# =============================================================================
# Module Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def embedder():
    """Shared embedder instance (session-scoped for performance)"""
    try:
        from src.modules import TextEmbedding
        return TextEmbedding(provider="local")
    except Exception as e:
        pytest.skip(f"TextEmbedding not available: {e}")


@pytest.fixture
def knowledge_base(temp_dir):
    """Create a temporary knowledge base"""
    try:
        from src.modules import KnowledgeBase
        return KnowledgeBase(base_dir=str(temp_dir))
    except Exception as e:
        pytest.skip(f"KnowledgeBase not available: {e}")


@pytest.fixture
def tts():
    """Create TTS instance"""
    try:
        from src.modules import TextToSpeech
        return TextToSpeech(voice="vi-female")
    except Exception as e:
        pytest.skip(f"TTS not available: {e}")


@pytest.fixture
def document_processor():
    """Create UnifiedProcessor instance"""
    try:
        from src.modules import UnifiedProcessor
        return UnifiedProcessor()
    except Exception as e:
        pytest.skip(f"UnifiedProcessor not available: {e}")


@pytest.fixture
def vector_db(embedder):
    """Create temporary vector database"""
    try:
        from src.modules import VectorDatabase
        db = VectorDatabase(
            collection_name="pytest_temp",
            embedding_dimension=embedder.embedding_dim
        )
        yield db
        db.delete_collection()
    except Exception as e:
        pytest.skip(f"VectorDatabase not available: {e}")
