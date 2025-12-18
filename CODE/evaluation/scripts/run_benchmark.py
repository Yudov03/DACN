"""
Run Benchmark Script - Chay benchmark voi dataset thuc
"""

import sys
import os
import io
import json
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add paths (from evaluation/scripts/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)


def run_benchmark(dataset_name: str = "vietnamese"):
    """
    Chay benchmark voi dataset

    Args:
        dataset_name: "vietnamese" hoac "squad"
    """
    from modules.embedding_module import TextEmbedding
    from modules.vector_db_module import VectorDatabase
    from modules.rag_module import RAGSystem
    from modules.chunking_module import TextChunker
    from modules.evaluation_module import RAGEvaluator

    print("=" * 70)
    print(f"RAG BENCHMARK - {dataset_name.upper()} Dataset")
    print("=" * 70)

    # Load dataset
    eval_dir = Path(__file__).parent.parent
    if dataset_name == "vietnamese":
        dataset_path = eval_dir / "datasets" / "vietnamese_sample.json"
    else:
        dataset_path = eval_dir / "datasets" / "squad_2_0_sample.json"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run: python scripts/download_dataset.py --dataset " + dataset_name)
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    contexts = dataset.get("contexts", [])
    test_cases = dataset.get("test_cases", [])

    print(f"\nDataset: {dataset.get('dataset', 'Unknown')}")
    print(f"Contexts: {len(contexts)}")
    print(f"Test cases: {len(test_cases)}")

    # Get provider (default to local)
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    print(f"Provider: {provider.upper()}")

    # Initialize components
    print("\n[1/5] Khoi tao Embedder...")
    embedder = TextEmbedding(provider=provider)

    print("\n[2/5] Khoi tao Chunker...")
    chunker = TextChunker(
        chunk_size=500,
        chunk_overlap=50,
        method="recursive"
    )

    print("\n[3/5] Khoi tao Vector DB...")
    dim = embedder.embedding_dim
    collection_name = f"benchmark_{dataset_name}_{int(datetime.now().timestamp())}"
    vector_db = VectorDatabase(
        collection_name=collection_name,
        embedding_dimension=dim
    )

    print("\n[4/5] Them contexts vao vector DB...")

    # Process contexts into chunks
    all_chunks = []
    for ctx in contexts:
        # Chunk each context
        chunks = chunker.chunk_text(
            ctx["text"],
            metadata={
                "doc_id": ctx["id"],
                "title": ctx.get("title", "")
            }
        )
        # Add context id to each chunk
        for chunk in chunks:
            chunk["context_id"] = ctx["id"]
        all_chunks.extend(chunks)

    print(f"  Total chunks: {len(all_chunks)}")

    # Encode and store
    chunks_with_emb = embedder.encode_chunks(all_chunks)
    vector_db.add_documents(chunks_with_emb)

    print("\n[5/5] Khoi tao RAG System...")
    rag = RAGSystem(
        vector_db=vector_db,
        embedder=embedder,
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        top_k=5
    )

    # Run evaluation
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    evaluator = RAGEvaluator(
        rag_system=rag,
        embedder=embedder,
        vector_db=vector_db
    )

    # Evaluate
    results = evaluator.evaluate_end_to_end(
        test_cases,
        k_values=[1, 3, 5, 10]
    )

    # Print detailed results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print("\n--- RETRIEVAL METRICS ---")
    ret_metrics = results["retrieval"]["average"]
    print(f"  MRR:           {ret_metrics['mrr']:.4f}")
    print(f"  Precision@1:   {ret_metrics['precision@1']:.4f}")
    print(f"  Precision@5:   {ret_metrics['precision@5']:.4f}")
    print(f"  Recall@5:      {ret_metrics['recall@5']:.4f}")
    print(f"  NDCG@5:        {ret_metrics['ndcg@5']:.4f}")
    print(f"  Hit Rate@5:    {ret_metrics['hit_rate@5']:.4f}")
    print(f"  Latency (ms):  {ret_metrics['latency_ms']:.2f}")

    print("\n--- GENERATION METRICS ---")
    gen_metrics = results["generation"]["average"]
    print(f"  Exact Match:   {gen_metrics['exact_match']:.4f}")
    print(f"  F1 Score:      {gen_metrics['f1_score']:.4f}")
    print(f"  BLEU Score:    {gen_metrics['bleu_score']:.4f}")
    print(f"  Semantic Sim:  {gen_metrics.get('semantic_similarity', 0):.4f}")
    print(f"  Latency (ms):  {gen_metrics['latency_ms']:.2f}")

    print("\n--- OVERALL ---")
    overall = results["overall"]
    print(f"  Retrieval MRR:     {overall['retrieval_mrr']:.4f}")
    print(f"  Generation F1:     {overall['generation_f1']:.4f}")
    print(f"  Semantic Score:    {overall['generation_semantic']:.4f}")
    print(f"  Avg Latency (ms):  {overall['avg_latency_ms']:.2f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_{dataset_name}_{timestamp}.json"

    full_results = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "provider": provider,
        "num_contexts": len(contexts),
        "num_chunks": len(all_chunks),
        "num_test_cases": len(test_cases),
        "config": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k": 5,
            "embedding_dim": dim
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Cleanup
    vector_db.delete_collection()
    print(f"Deleted collection: {collection_name}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)

    return results


def compare_configs(dataset_name: str = "vietnamese"):
    """So sanh nhieu cau hinh khac nhau"""

    configs = [
        {"chunk_size": 300, "chunk_overlap": 30, "top_k": 3},
        {"chunk_size": 500, "chunk_overlap": 50, "top_k": 5},
        {"chunk_size": 800, "chunk_overlap": 100, "top_k": 10},
    ]

    print("=" * 70)
    print("CONFIG COMPARISON")
    print("=" * 70)

    results = []
    for i, config in enumerate(configs):
        print(f"\n[Config {i+1}/{len(configs)}] {config}")
        # Would run benchmark with each config...
        # results.append(run_benchmark_with_config(dataset_name, config))

    print("\nNote: Full config comparison requires running multiple benchmarks")
    print("Use tune_parameters.py for automated parameter tuning")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG Benchmark")
    parser.add_argument(
        "--dataset",
        choices=["vietnamese", "squad"],
        default="vietnamese",
        help="Dataset to use"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple configs"
    )

    args = parser.parse_args()

    if args.compare:
        compare_configs(args.dataset)
    else:
        run_benchmark(args.dataset)
