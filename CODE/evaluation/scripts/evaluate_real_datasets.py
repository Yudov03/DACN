"""
Evaluate Audio IR System with Real Datasets
- Vietnamese QA Dataset
- SQuAD 2.0 Dataset

So sanh cac phuong phap:
- Vector Search (Semantic)
- BM25 Search (Keyword)
- Hybrid Search (Vector + BM25)
"""

import sys
import os
import io
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths (from evaluation/scripts/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)


class DatasetEvaluator:
    """Evaluator cho cac datasets thuc te"""

    def __init__(
        self,
        embedding_provider: str = "local",
        embedding_model: str = "sbert",
        use_reranker: bool = False
    ):
        from modules.embedding_module import TextEmbedding
        from modules.vector_db_module import VectorDatabase
        from modules.evaluation_module import RAGEvaluator

        print("=" * 70)
        print("INITIALIZING EVALUATION SYSTEM")
        print("=" * 70)

        # Initialize embedding
        print(f"\n[1/3] Loading embedding model ({embedding_provider}: {embedding_model})...")
        self.embedder = TextEmbedding(
            provider=embedding_provider,
            model_name=embedding_model
        )

        # Vector DB will be created per dataset
        self.vector_db = None
        self.evaluator = RAGEvaluator(embedder=self.embedder)

        # Reranker
        self.reranker = None
        if use_reranker:
            print("\n[2/3] Loading reranker model...")
            try:
                from modules.reranker_module import CrossEncoderReranker
                self.reranker = CrossEncoderReranker()
            except Exception as e:
                print(f"Warning: Could not load reranker: {e}")

        print("\n[3/3] System ready!")

    def load_dataset(self, dataset_path: str) -> Dict:
        """Load dataset from JSON file"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def index_contexts(self, contexts: List[Dict], collection_name: str):
        """Index contexts vao vector database"""
        from modules.vector_db_module import VectorDatabase

        print(f"\nIndexing {len(contexts)} contexts...")

        # Create vector db
        self.vector_db = VectorDatabase(
            collection_name=collection_name,
            embedding_dimension=self.embedder.embedding_dim
        )

        # Prepare chunks
        chunks = []
        for ctx in contexts:
            chunk = {
                "text": ctx["text"],
                "chunk_id": ctx["id"],
                "title": ctx.get("title", ""),
            }
            chunks.append(chunk)

        # Encode
        texts = [c["text"] for c in chunks]
        print("Encoding contexts...")
        embeddings = self.embedder.encode_text(texts, show_progress=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()

        # Add to vector db
        self.vector_db.add_documents(chunks)
        print(f"Indexed {len(chunks)} contexts")

    def evaluate_retrieval(
        self,
        test_cases: List[Dict],
        search_method: str = "vector",
        top_k_values: List[int] = [1, 3, 5, 10],
        alpha: float = 0.5
    ) -> Dict:
        """
        Evaluate retrieval performance

        Args:
            test_cases: List of test cases with query and relevant_doc_ids
            search_method: "vector", "bm25", or "hybrid"
            top_k_values: List of k values to evaluate
            alpha: Weight for vector search in hybrid (0-1)

        Returns:
            Dict with metrics
        """
        results = {
            "method": search_method,
            "num_queries": len(test_cases),
            "metrics": {}
        }

        # Initialize metrics
        for k in top_k_values:
            results["metrics"][f"precision@{k}"] = []
            results["metrics"][f"recall@{k}"] = []
            results["metrics"][f"ndcg@{k}"] = []
            results["metrics"][f"hit_rate@{k}"] = []
        results["metrics"]["mrr"] = []
        results["metrics"]["latency_ms"] = []

        max_k = max(top_k_values)

        for test_case in test_cases:
            query = test_case["query"]
            relevant_ids = test_case.get("relevant_doc_ids", [])

            # Search
            start_time = time.time()

            if search_method == "vector":
                query_emb = self.embedder.encode_query(query)
                retrieved = self.vector_db.search(query_emb, top_k=max_k)

            elif search_method == "bm25":
                # BM25 only (via hybrid with alpha=0)
                query_emb = self.embedder.encode_query(query)
                retrieved = self.vector_db.hybrid_search(
                    query=query,
                    query_embedding=query_emb,
                    top_k=max_k,
                    alpha=0.0
                )

            elif search_method == "hybrid":
                query_emb = self.embedder.encode_query(query)
                retrieved = self.vector_db.hybrid_search(
                    query=query,
                    query_embedding=query_emb,
                    top_k=max_k,
                    alpha=alpha
                )

            elif search_method == "rerank" and self.reranker:
                query_emb = self.embedder.encode_query(query)
                retrieved = self.vector_db.search_with_rerank(
                    query=query,
                    query_embedding=query_emb,
                    reranker=self.reranker,
                    top_k=max_k,
                    fetch_k=max_k * 2
                )
            else:
                query_emb = self.embedder.encode_query(query)
                retrieved = self.vector_db.search(query_emb, top_k=max_k)

            latency = (time.time() - start_time) * 1000

            # Get retrieved IDs
            retrieved_ids = [r.get("metadata", {}).get("chunk_id", r.get("id", "")) for r in retrieved]

            # Calculate metrics
            for k in top_k_values:
                results["metrics"][f"precision@{k}"].append(
                    self.evaluator.precision_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"recall@{k}"].append(
                    self.evaluator.recall_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"ndcg@{k}"].append(
                    self.evaluator.ndcg_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"hit_rate@{k}"].append(
                    self.evaluator.hit_rate_at_k(retrieved_ids, relevant_ids, k)
                )

            results["metrics"]["mrr"].append(
                self.evaluator.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            )
            results["metrics"]["latency_ms"].append(latency)

        # Calculate averages
        results["average"] = {}
        for metric_name, values in results["metrics"].items():
            results["average"][metric_name] = sum(values) / len(values) if values else 0

        return results

    def run_full_evaluation(
        self,
        dataset_path: str,
        dataset_name: str = "dataset"
    ) -> Dict:
        """Run full evaluation on a dataset"""
        print("\n" + "=" * 70)
        print(f"EVALUATING: {dataset_name}")
        print("=" * 70)

        # Load dataset
        print("\nLoading dataset...")
        dataset = self.load_dataset(dataset_path)
        contexts = dataset.get("contexts", [])
        test_cases = dataset.get("test_cases", [])

        print(f"  Contexts: {len(contexts)}")
        print(f"  Test cases: {len(test_cases)}")

        # Index contexts
        collection_name = f"eval_{dataset_name}_{int(time.time())}"
        self.index_contexts(contexts, collection_name)

        # Evaluate different methods
        results = {
            "dataset": dataset_name,
            "num_contexts": len(contexts),
            "num_queries": len(test_cases),
            "timestamp": datetime.now().isoformat(),
            "embedding_model": self.embedder.model_name,
            "methods": {}
        }

        # 1. Vector Search
        print("\n--- Evaluating Vector Search ---")
        results["methods"]["vector"] = self.evaluate_retrieval(
            test_cases, search_method="vector"
        )

        # 2. BM25 Search
        print("\n--- Evaluating BM25 Search ---")
        results["methods"]["bm25"] = self.evaluate_retrieval(
            test_cases, search_method="bm25"
        )

        # 3. Hybrid Search (alpha=0.5)
        print("\n--- Evaluating Hybrid Search (alpha=0.5) ---")
        results["methods"]["hybrid_0.5"] = self.evaluate_retrieval(
            test_cases, search_method="hybrid", alpha=0.5
        )

        # 4. Hybrid Search (alpha=0.7)
        print("\n--- Evaluating Hybrid Search (alpha=0.7) ---")
        results["methods"]["hybrid_0.7"] = self.evaluate_retrieval(
            test_cases, search_method="hybrid", alpha=0.7
        )

        # 5. With Reranker (if available)
        if self.reranker:
            print("\n--- Evaluating with Reranker ---")
            results["methods"]["rerank"] = self.evaluate_retrieval(
                test_cases, search_method="rerank"
            )

        # Cleanup
        self.vector_db.delete_collection()

        return results

    def print_comparison(self, results: Dict):
        """Print comparison table"""
        print("\n" + "=" * 70)
        print("RESULTS COMPARISON")
        print("=" * 70)

        methods = results.get("methods", {})
        metrics_to_show = ["mrr", "precision@1", "precision@5", "recall@5", "ndcg@5", "hit_rate@5", "latency_ms"]

        # Header
        print(f"\n{'Metric':<15}", end="")
        for method in methods.keys():
            print(f"{method:<15}", end="")
        print()
        print("-" * (15 + 15 * len(methods)))

        # Metrics
        for metric in metrics_to_show:
            print(f"{metric:<15}", end="")
            for method, data in methods.items():
                value = data.get("average", {}).get(metric, 0)
                if metric == "latency_ms":
                    print(f"{value:<15.2f}", end="")
                else:
                    print(f"{value:<15.4f}", end="")
            print()

        # Best method
        print("\n" + "-" * 50)
        mrr_values = {m: d["average"]["mrr"] for m, d in methods.items()}
        best_method = max(mrr_values, key=mrr_values.get)
        print(f"Best method (by MRR): {best_method} ({mrr_values[best_method]:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate with real datasets")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["vietnamese", "squad", "all"],
                       help="Dataset to evaluate")
    parser.add_argument("--embedding", type=str, default="sbert",
                       choices=["sbert", "e5", "e5-large"],
                       help="Embedding model")
    parser.add_argument("--reranker", action="store_true",
                       help="Use cross-encoder reranker")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = DatasetEvaluator(
        embedding_provider="local",
        embedding_model=args.embedding,
        use_reranker=args.reranker
    )

    # Dataset paths
    data_dir = Path(__file__).parent.parent / "datasets"
    datasets = {}

    if args.dataset in ["vietnamese", "all"]:
        vn_path = data_dir / "vietnamese_sample.json"
        if vn_path.exists():
            datasets["vietnamese"] = str(vn_path)

    if args.dataset in ["squad", "all"]:
        squad_path = data_dir / "squad_2_0_sample.json"
        if squad_path.exists():
            datasets["squad"] = str(squad_path)

    if not datasets:
        print("No datasets found!")
        return

    # Run evaluations
    all_results = {}

    for name, path in datasets.items():
        results = evaluator.run_full_evaluation(path, name)
        all_results[name] = results
        evaluator.print_comparison(results)

    # Save results
    if args.save:
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"eval_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        methods = results.get("methods", {})
        for method, data in methods.items():
            mrr = data["average"]["mrr"]
            ndcg5 = data["average"]["ndcg@5"]
            latency = data["average"]["latency_ms"]
            print(f"  {method:<15} MRR={mrr:.4f}  NDCG@5={ndcg5:.4f}  Latency={latency:.1f}ms")


if __name__ == "__main__":
    main()
