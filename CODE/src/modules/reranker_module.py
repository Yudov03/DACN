"""
Reranker Module - Cai thien thu tu ket qua retrieval
Ho tro:
- Cross-encoder reranking (sentence-transformers)
- LLM-based reranking
- Simple similarity reranking
"""

from typing import List, Optional, Tuple
import numpy as np


class CrossEncoderReranker:
    """
    Reranker su dung Cross-Encoder model tu sentence-transformers
    Cross-encoder cho ket qua tot hon bi-encoder cho reranking
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Khoi tao Cross-Encoder Reranker

        Args:
            model_name: Ten model cross-encoder
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (nhanh, tot)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (tot hon)
                - BAAI/bge-reranker-base (multilingual)
                - BAAI/bge-reranker-large (best quality)
            device: "cuda" hoac "cpu" (None = auto)
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers chua duoc cai dat. "
                "Chay: pip install sentence-transformers"
            )

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Cross-Encoder reranker: {model_name} on {device}...")
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        print("Reranker loaded successfully!")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents cho query

        Args:
            query: Query text
            documents: List document texts
            top_k: So luong ket qua (None = all)

        Returns:
            List of (original_index, score) sorted by score descending
        """
        if not documents:
            return []

        # Create pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores
        scores = self.model.predict(pairs)

        # Create ranked list
        ranked = [(i, float(score)) for i, score in enumerate(scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            ranked = ranked[:top_k]

        return ranked

    def __call__(self, query: str, documents: List[str]) -> List[float]:
        """Callable interface for VectorDatabase.search_with_rerank"""
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        return [float(s) for s in scores]


class EmbeddingReranker:
    """
    Reranker don gian su dung embedding similarity
    Dung khi khong co Cross-Encoder model
    """

    def __init__(self, embedder):
        """
        Args:
            embedder: TextEmbedding instance
        """
        self.embedder = embedder

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents dua tren embedding similarity

        Args:
            query: Query text
            documents: List document texts
            top_k: So luong ket qua

        Returns:
            List of (original_index, score)
        """
        if not documents:
            return []

        # Encode query
        query_emb = self.embedder.encode_query(query)

        # Encode documents
        doc_embs = self.embedder.encode_text(documents, show_progress=False)

        # Calculate similarities
        scores = []
        for doc_emb in doc_embs:
            sim = self.embedder.compute_similarity(query_emb, doc_emb)
            scores.append(float(sim))

        # Create ranked list
        ranked = [(i, score) for i, score in enumerate(scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            ranked = ranked[:top_k]

        return ranked

    def __call__(self, query: str, documents: List[str]) -> List[float]:
        """Callable interface"""
        if not documents:
            return []

        query_emb = self.embedder.encode_query(query)
        doc_embs = self.embedder.encode_text(documents, show_progress=False)

        scores = []
        for doc_emb in doc_embs:
            sim = self.embedder.compute_similarity(query_emb, doc_emb)
            scores.append(float(sim))

        return scores


class LLMReranker:
    """
    Reranker su dung LLM de danh gia relevance
    Cham nhung co the cho ket qua tot cho cau hoi phuc tap
    """

    def __init__(self, llm, prompt_template: Optional[str] = None):
        """
        Args:
            llm: LLM instance (ChatOpenAI, ChatOllama, etc.)
            prompt_template: Custom prompt template
        """
        self.llm = llm
        self.prompt_template = prompt_template or """
Rate the relevance of the following document to the query on a scale of 0-10.
Only respond with a number.

Query: {query}

Document: {document}

Relevance score (0-10):"""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank using LLM"""
        if not documents:
            return []

        scores = []
        for doc in documents:
            prompt = self.prompt_template.format(query=query, document=doc[:500])
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content.strip()
                # Extract number
                import re
                match = re.search(r'\d+\.?\d*', score_text)
                score = float(match.group()) / 10 if match else 0.5
                scores.append(min(max(score, 0), 1))
            except Exception as e:
                print(f"LLM rerank error: {e}")
                scores.append(0.5)

        ranked = [(i, score) for i, score in enumerate(scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            ranked = ranked[:top_k]

        return ranked

    def __call__(self, query: str, documents: List[str]) -> List[float]:
        """Callable interface"""
        results = self.rerank(query, documents)
        # Restore original order for scores
        scores = [0.0] * len(documents)
        for idx, score in results:
            scores[idx] = score
        return scores


# Convenience function to create reranker
def create_reranker(
    method: str = "cross-encoder",
    model_name: Optional[str] = None,
    embedder=None,
    llm=None
):
    """
    Factory function to create reranker

    Args:
        method: "cross-encoder", "embedding", or "llm"
        model_name: Model name for cross-encoder
        embedder: TextEmbedding instance (for embedding reranker)
        llm: LLM instance (for llm reranker)

    Returns:
        Reranker instance
    """
    if method == "cross-encoder":
        model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return CrossEncoderReranker(model_name=model_name)
    elif method == "embedding":
        if embedder is None:
            raise ValueError("embedder required for embedding reranker")
        return EmbeddingReranker(embedder)
    elif method == "llm":
        if llm is None:
            raise ValueError("llm required for llm reranker")
        return LLMReranker(llm)
    else:
        raise ValueError(f"Unknown reranker method: {method}")


# Test
if __name__ == "__main__":
    print("Testing Reranker Module...")

    # Test data
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny and warm.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language.",
        "ML algorithms learn patterns from data."
    ]

    # Test Cross-Encoder reranker
    print("\n--- Cross-Encoder Reranker ---")
    try:
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, documents)
        print(f"Query: {query}")
        for idx, score in results:
            print(f"  [{score:.4f}] {documents[idx][:50]}...")
    except Exception as e:
        print(f"Error: {e}")

    print("\nReranker Module initialized successfully!")
