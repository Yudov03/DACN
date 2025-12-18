"""
Demo script for RAG Pipeline
=============================

Demonstrates the complete RAG pipeline:
- Text Chunking
- Embedding (Local/Cloud)
- Vector Database (Qdrant + BM25)
- Hybrid Search
- LLM Generation

Usage:
    python scripts/demo_rag_pipeline.py
    python scripts/demo_rag_pipeline.py --provider local
    python scripts/demo_rag_pipeline.py --provider google
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demo_chunking():
    """Demo text chunking."""
    print_section("1. TEXT CHUNKING")

    from modules import TextChunker

    text = """
    Machine Learning là một nhánh của Trí tuệ Nhân tạo (AI).
    Nó cho phép máy tính học từ dữ liệu mà không cần lập trình tường minh.

    Có ba loại Machine Learning chính:
    1. Supervised Learning - Học có giám sát
    2. Unsupervised Learning - Học không giám sát
    3. Reinforcement Learning - Học tăng cường

    Deep Learning là một dạng đặc biệt của Machine Learning,
    sử dụng mạng neural nhiều tầng để học các biểu diễn phức tạp.
    """

    print("\n--- Fixed Size Chunking ---")
    chunker = TextChunker(chunk_size=200, chunk_overlap=20, method="fixed")
    chunks = chunker.chunk_text(text.strip(), metadata={"source": "demo"})
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1} ({len(chunk['text'])} chars):")
        print(f"  {chunk['text'][:100]}...")

    print("\n--- Sentence-based Chunking ---")
    chunker = TextChunker(chunk_size=200, method="sentence")
    chunks = chunker.chunk_text(text.strip())
    print(f"Created {len(chunks)} chunks")

    print("\n--- Recursive Chunking ---")
    chunker = TextChunker(chunk_size=200, method="recursive")
    chunks = chunker.chunk_text(text.strip())
    print(f"Created {len(chunks)} chunks")


def demo_embedding(provider: str = "local"):
    """Demo text embedding."""
    print_section(f"2. TEXT EMBEDDING ({provider.upper()})")

    from modules import TextEmbedding

    embedder = TextEmbedding(provider=provider)
    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.embedding_dim}")

    # Single text
    print("\n--- Single Text Embedding ---")
    text = "Machine learning là gì?"
    embedding = embedder.encode_text(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

    # Multiple texts
    print("\n--- Batch Embedding ---")
    texts = [
        "AI là trí tuệ nhân tạo",
        "Deep learning sử dụng neural networks",
        "NLP xử lý ngôn ngữ tự nhiên"
    ]
    embeddings = embedder.encode_text(texts)
    print(f"Texts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Similarity
    print("\n--- Similarity ---")
    emb1 = embedder.encode_text("Machine learning")
    emb2 = embedder.encode_text("Deep learning")
    emb3 = embedder.encode_text("Cooking recipes")

    sim_related = embedder.compute_similarity(emb1, emb2)
    sim_unrelated = embedder.compute_similarity(emb1, emb3)
    print(f"ML vs DL: {sim_related:.4f}")
    print(f"ML vs Cooking: {sim_unrelated:.4f}")

    return embedder


def demo_vector_db(embedder):
    """Demo vector database."""
    print_section("3. VECTOR DATABASE")

    from modules import VectorDatabase

    # Create temporary collection
    collection_name = "demo_rag_test"
    db = VectorDatabase(
        collection_name=collection_name,
        embedding_dimension=embedder.embedding_dim
    )

    try:
        # Add documents
        print("\n--- Adding Documents ---")
        docs = [
            {"text": "Machine Learning cho phép máy tính học từ dữ liệu", "topic": "ML"},
            {"text": "Deep Learning sử dụng neural networks nhiều tầng", "topic": "DL"},
            {"text": "NLP giúp máy tính hiểu ngôn ngữ con người", "topic": "NLP"},
            {"text": "Computer Vision xử lý và phân tích hình ảnh", "topic": "CV"},
            {"text": "Reinforcement Learning học thông qua phần thưởng", "topic": "RL"},
        ]

        chunks = []
        for i, doc in enumerate(docs):
            embedding = embedder.encode_text(doc["text"])
            chunks.append({
                "id": str(i),
                "text": doc["text"],
                "embedding": embedding.tolist(),
                "metadata": {"topic": doc["topic"]}
            })

        db.add_documents(chunks)
        print(f"Added {len(chunks)} documents")

        # Vector search
        print("\n--- Vector Search ---")
        query = "Học máy là gì?"
        query_emb = embedder.encode_query(query)
        results = db.search(query_embedding=query_emb, top_k=3)

        print(f"Query: {query}")
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['similarity']:.3f}] {r['text'][:50]}...")

        # Hybrid search
        print("\n--- Hybrid Search (Vector + BM25) ---")
        results = db.hybrid_search(
            query=query,
            query_embedding=query_emb,
            alpha=0.7,
            top_k=3
        )

        print(f"Query: {query}")
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['similarity']:.3f}] {r['text'][:50]}...")

        # Stats
        print("\n--- Collection Stats ---")
        stats = db.get_collection_stats()
        print(f"Documents: {stats.get('count', 0)}")

    finally:
        # Cleanup
        db.delete_collection()
        print(f"\nDeleted collection: {collection_name}")


def demo_rag_query(provider: str = "local"):
    """Demo RAG query."""
    print_section("4. RAG QUERY")

    from modules import TextEmbedding, VectorDatabase, RAGSystem
    import os

    # Setup
    embedder = TextEmbedding(provider=provider)
    collection_name = "demo_rag_query"
    db = VectorDatabase(
        collection_name=collection_name,
        embedding_dimension=embedder.embedding_dim
    )

    try:
        # Add knowledge
        print("\n--- Building Knowledge Base ---")
        knowledge = [
            "Học phí năm 2024 là 15 triệu đồng mỗi kỳ.",
            "Sinh viên cần đạt tối thiểu 80% điểm danh.",
            "Điểm thi cuối kỳ chiếm 60% tổng điểm.",
            "Điểm giữa kỳ chiếm 40% tổng điểm.",
            "Thời gian đăng ký môn học là 2 tuần đầu học kỳ.",
        ]

        chunks = []
        for i, text in enumerate(knowledge):
            embedding = embedder.encode_text(text)
            chunks.append({
                "id": str(i),
                "text": text,
                "embedding": embedding.tolist()
            })

        db.add_documents(chunks)
        print(f"Added {len(knowledge)} knowledge items")

        # Create RAG system
        llm_provider = os.getenv("LLM_PROVIDER", "ollama")
        print(f"\n--- RAG System ({llm_provider}) ---")

        rag = RAGSystem(
            vector_db=db,
            embedder=embedder,
            provider=llm_provider,
            top_k=3
        )

        # Query
        questions = [
            "Học phí một kỳ là bao nhiêu?",
            "Điểm thi cuối kỳ chiếm bao nhiêu phần trăm?",
        ]

        for question in questions:
            print(f"\nQ: {question}")
            try:
                response = rag.query(question, top_k=3)
                print(f"A: {response.get('answer', 'No answer')}")
            except Exception as e:
                print(f"Error: {e}")

    finally:
        db.delete_collection()
        print(f"\nDeleted collection: {collection_name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo RAG Pipeline")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "google", "openai"],
        default="local",
        help="Embedding provider"
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG query demo (requires LLM)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("   RAG PIPELINE DEMO")
    print("=" * 60)

    # Run demos
    demo_chunking()
    embedder = demo_embedding(args.provider)
    demo_vector_db(embedder)

    if not args.skip_rag:
        demo_rag_query(args.provider)

    print("\n" + "=" * 60)
    print("   DEMO COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
