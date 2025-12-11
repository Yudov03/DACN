"""
Modules package - Chua cac module chinh cua he thong
"""

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
    'get_rag_prompt'
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
