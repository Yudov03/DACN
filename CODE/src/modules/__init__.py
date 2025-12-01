"""
Modules package - Chứa các module chính của hệ thống
"""

# Lazy imports to avoid dependency errors
__all__ = [
    'WhisperASR',
    'TextChunker',
    'TextEmbedding',
    'VectorDatabase',
    'RAGSystem'
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
