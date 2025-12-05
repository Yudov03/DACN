"""
Modules package - Chua cac module chinh cua he thong
"""

# Lazy imports to avoid dependency errors
__all__ = [
    'WhisperASR',
    'TextChunker',
    'TextEmbedding',
    'VectorDatabase',
    'RAGSystem',
    'RAGEvaluator'
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
    elif name == 'RAGSystem':
        from .rag_module import RAGSystem
        return RAGSystem
    elif name == 'RAGEvaluator':
        from .evaluation_module import RAGEvaluator
        return RAGEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
