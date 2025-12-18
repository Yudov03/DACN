"""
Modules package - Chua cac module chinh cua he thong
"""

# ============================================================================
# LangChain 0.x -> 1.x Compatibility Shim for PaddleX
# ============================================================================
# PaddleX (PaddleOCR dependency) uses old langchain imports:
#   from langchain.docstore.document import Document
#   from langchain.text_splitter import RecursiveCharacterTextSplitter
# LangChain 1.x moved these to langchain_core and langchain_text_splitters
# This shim creates fake modules to redirect the imports.
# See: https://github.com/PaddlePaddle/PaddleX/issues/4765
# ============================================================================
import sys
import types

def _setup_langchain_compat():
    """Create compatibility shim for langchain 0.x imports -> langchain 1.x"""
    # Skip if already set up
    if 'langchain.docstore' in sys.modules:
        return

    try:
        # Import from new locations
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Create fake langchain.docstore module
        docstore_module = types.ModuleType('langchain.docstore')
        docstore_module.__path__ = []

        # Create fake langchain.docstore.document submodule
        document_module = types.ModuleType('langchain.docstore.document')
        document_module.Document = Document

        # Create fake langchain.text_splitter module
        text_splitter_module = types.ModuleType('langchain.text_splitter')
        text_splitter_module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

        # Register in sys.modules
        sys.modules['langchain.docstore'] = docstore_module
        sys.modules['langchain.docstore.document'] = document_module
        sys.modules['langchain.text_splitter'] = text_splitter_module

        # Add document as attribute of docstore
        docstore_module.document = document_module

    except ImportError:
        # langchain_core or langchain_text_splitters not installed, skip
        pass

# Apply the shim on module load
_setup_langchain_compat()

# Lazy imports to avoid dependency errors
__all__ = [
    'WhisperASR',
    'TextChunker',
    'TextEmbedding',
    'VectorDatabase',
    'BM25',
    'RAGSystem',
    'RAGEvaluator',
    'CrossEncoderReranker',
    'EmbeddingReranker',
    'create_reranker',
    # Phase 1 Optimization modules
    'QueryExpander',
    'MultiQueryRetriever',
    'ContextCompressor',
    'ContextualCompressor',
    'CacheManager',
    'EmbeddingCache',
    'ResponseCache',
    'PromptTemplateManager',
    'get_rag_prompt',
    # Document Processing (Phase 1 - RAG Platform)
    'UnifiedProcessor',
    'PDFProcessor',
    'WordProcessor',
    'ExcelProcessor',
    'PowerPointProcessor',
    'ImageProcessor',
    'TextProcessor',
    'AudioProcessor',
    'VideoProcessor',
    'ProcessedDocument',
    'process_document',
    'format_transcript_with_timestamps',
    # Knowledge Base (Phase 2 - RAG Platform)
    'KnowledgeBase',
    'DocumentInfo',
    'KBStats',
    # TTS (Phase 4 - RAG Platform)
    'TextToSpeech',
    'text_to_speech',
    'text_to_audio_bytes',
    # Anti-Hallucination & Quality (Enhanced RAG)
    'AnswerVerifier',
    'AbstentionChecker',
    'VerificationResult',
    'verify_rag_answer',
    'should_abstain_from_answering',
    # Conflict Detection (Enhanced RAG)
    'ConflictDetector',
    'ConflictInfo',
    'ConflictDetectionResult',
    'DateAwareRetriever',
    'detect_conflicts',
    'resolve_version_conflicts',
    # Post-Processing (Vietnamese text correction)
    'PostProcessor',
    'post_process_text',
    # File Types (centralized definitions)
    'AUDIO_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'DOCUMENT_EXTENSIONS',
    'ALL_SUPPORTED_EXTENSIONS',
    'MEDIA_TYPES',
    'TYPE_TO_FOLDER',
    'get_file_category',
    'get_folder_for_type',
    'is_media_file',
]

def __getattr__(name):
    """Lazy import modules"""
    if name == 'WhisperASR':
        from .asr_module import WhisperASR
        return WhisperASR
    elif name == 'TextChunker':
        from .chunking_module import TextChunker
        return TextChunker
    elif name == 'TextEmbedding':
        from .embedding_module import TextEmbedding
        return TextEmbedding
    elif name == 'VectorDatabase':
        from .vector_db_module import VectorDatabase
        return VectorDatabase
    elif name == 'BM25':
        from .vector_db_module import BM25
        return BM25
    elif name == 'RAGSystem':
        from .rag_module import RAGSystem
        return RAGSystem
    elif name == 'RAGEvaluator':
        from .evaluation_module import RAGEvaluator
        return RAGEvaluator
    elif name == 'CrossEncoderReranker':
        from .reranker_module import CrossEncoderReranker
        return CrossEncoderReranker
    elif name == 'EmbeddingReranker':
        from .reranker_module import EmbeddingReranker
        return EmbeddingReranker
    elif name == 'create_reranker':
        from .reranker_module import create_reranker
        return create_reranker
    # Query Expansion
    elif name == 'QueryExpander':
        from .query_expansion_module import QueryExpander
        return QueryExpander
    elif name == 'MultiQueryRetriever':
        from .query_expansion_module import MultiQueryRetriever
        return MultiQueryRetriever
    # Context Compression
    elif name == 'ContextCompressor':
        from .context_compression_module import ContextCompressor
        return ContextCompressor
    elif name == 'ContextualCompressor':
        from .context_compression_module import ContextualCompressor
        return ContextualCompressor
    # Caching
    elif name == 'CacheManager':
        from .caching_module import CacheManager
        return CacheManager
    elif name == 'EmbeddingCache':
        from .caching_module import EmbeddingCache
        return EmbeddingCache
    elif name == 'ResponseCache':
        from .caching_module import ResponseCache
        return ResponseCache
    # Prompt Templates
    elif name == 'PromptTemplateManager':
        from .prompt_templates import PromptTemplateManager
        return PromptTemplateManager
    elif name == 'get_rag_prompt':
        from .prompt_templates import get_rag_prompt
        return get_rag_prompt
    # Document Processing
    elif name == 'UnifiedProcessor':
        from .document_processor import UnifiedProcessor
        return UnifiedProcessor
    elif name == 'PDFProcessor':
        from .document_processor import PDFProcessor
        return PDFProcessor
    elif name == 'WordProcessor':
        from .document_processor import WordProcessor
        return WordProcessor
    elif name == 'ExcelProcessor':
        from .document_processor import ExcelProcessor
        return ExcelProcessor
    elif name == 'PowerPointProcessor':
        from .document_processor import PowerPointProcessor
        return PowerPointProcessor
    elif name == 'ImageProcessor':
        from .document_processor import ImageProcessor
        return ImageProcessor
    elif name == 'TextProcessor':
        from .document_processor import TextProcessor
        return TextProcessor
    elif name == 'AudioProcessor':
        from .document_processor import AudioProcessor
        return AudioProcessor
    elif name == 'VideoProcessor':
        from .document_processor import VideoProcessor
        return VideoProcessor
    elif name == 'ProcessedDocument':
        from .document_processor import ProcessedDocument
        return ProcessedDocument
    elif name == 'process_document':
        from .document_processor import process_document
        return process_document
    elif name == 'format_transcript_with_timestamps':
        from .document_processor import format_transcript_with_timestamps
        return format_transcript_with_timestamps
    # Knowledge Base
    elif name == 'KnowledgeBase':
        from .knowledge_base import KnowledgeBase
        return KnowledgeBase
    elif name == 'DocumentInfo':
        from .knowledge_base import DocumentInfo
        return DocumentInfo
    elif name == 'KBStats':
        from .knowledge_base import KBStats
        return KBStats
    # TTS
    elif name == 'TextToSpeech':
        from .tts_module import TextToSpeech
        return TextToSpeech
    elif name == 'text_to_speech':
        from .tts_module import text_to_speech
        return text_to_speech
    elif name == 'text_to_audio_bytes':
        from .tts_module import text_to_audio_bytes
        return text_to_audio_bytes
    # Answer Verification (Anti-Hallucination)
    elif name == 'AnswerVerifier':
        from .answer_verification import AnswerVerifier
        return AnswerVerifier
    elif name == 'AbstentionChecker':
        from .answer_verification import AbstentionChecker
        return AbstentionChecker
    elif name == 'VerificationResult':
        from .answer_verification import VerificationResult
        return VerificationResult
    elif name == 'verify_rag_answer':
        from .answer_verification import verify_rag_answer
        return verify_rag_answer
    elif name == 'should_abstain_from_answering':
        from .answer_verification import should_abstain_from_answering
        return should_abstain_from_answering
    # Conflict Detection
    elif name == 'ConflictDetector':
        from .conflict_detection import ConflictDetector
        return ConflictDetector
    elif name == 'ConflictInfo':
        from .conflict_detection import ConflictInfo
        return ConflictInfo
    elif name == 'ConflictDetectionResult':
        from .conflict_detection import ConflictDetectionResult
        return ConflictDetectionResult
    elif name == 'DateAwareRetriever':
        from .conflict_detection import DateAwareRetriever
        return DateAwareRetriever
    elif name == 'detect_conflicts':
        from .conflict_detection import detect_conflicts
        return detect_conflicts
    elif name == 'resolve_version_conflicts':
        from .conflict_detection import resolve_version_conflicts
        return resolve_version_conflicts
    # Post-Processing
    elif name == 'PostProcessor':
        from .post_processing import PostProcessor
        return PostProcessor
    elif name == 'post_process_text':
        from .post_processing import post_process_text
        return post_process_text
    # File Types (centralized definitions)
    elif name == 'AUDIO_EXTENSIONS':
        from .file_types import AUDIO_EXTENSIONS
        return AUDIO_EXTENSIONS
    elif name == 'VIDEO_EXTENSIONS':
        from .file_types import VIDEO_EXTENSIONS
        return VIDEO_EXTENSIONS
    elif name == 'DOCUMENT_EXTENSIONS':
        from .file_types import DOCUMENT_EXTENSIONS
        return DOCUMENT_EXTENSIONS
    elif name == 'ALL_SUPPORTED_EXTENSIONS':
        from .file_types import ALL_SUPPORTED_EXTENSIONS
        return ALL_SUPPORTED_EXTENSIONS
    elif name == 'MEDIA_TYPES':
        from .file_types import MEDIA_TYPES
        return MEDIA_TYPES
    elif name == 'TYPE_TO_FOLDER':
        from .file_types import TYPE_TO_FOLDER
        return TYPE_TO_FOLDER
    elif name == 'get_file_category':
        from .file_types import get_file_category
        return get_file_category
    elif name == 'get_folder_for_type':
        from .file_types import get_folder_for_type
        return get_folder_for_type
    elif name == 'is_media_file':
        from .file_types import is_media_file
        return is_media_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
