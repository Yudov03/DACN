"""
Parameter Tuning Script - Tim cau hinh toi uu cho he thong RAG

Tham so co the tune:
- chunk_size: 200, 300, 500, 800, 1000
- chunk_overlap: 0, 20, 50, 100
- chunking_method: fixed, sentence, recursive
- top_k: 3, 5, 10, 15
- similarity_threshold: 0.3, 0.5, 0.7
- llm_temperature: 0.0, 0.3, 0.5, 0.7, 1.0
"""

import sys
import os
import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# Add paths (from evaluation/scripts/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)


class ParameterTuner:
    """
    Class de tune cac tham so cua he thong RAG
    """

    def __init__(
        self,
        test_data_path: str,
        output_dir: str = None
    ):
        """
        Khoi tao tuner

        Args:
            test_data_path: Duong dan toi file test data JSON
            output_dir: Thu muc luu ket qua
        """
        self.test_data_path = Path(test_data_path)
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "tuning_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load test data
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)

        print(f"Da load {len(self.test_data)} test cases")

        # Default parameter ranges
        self.param_ranges = {
            "chunk_size": [300, 500, 800],
            "chunk_overlap": [30, 50, 100],
            "chunking_method": ["recursive", "sentence"],
            "top_k": [3, 5, 10],
            "llm_temperature": [0.3, 0.7]
        }

        self.results = []

    def set_param_ranges(self, param_ranges: Dict[str, List]):
        """Cap nhat parameter ranges"""
        self.param_ranges.update(param_ranges)

    def _create_pipeline(self, params: Dict) -> Any:
        """
        Tao pipeline voi cac params cu the

        Args:
            params: Dict chua cac parameter values

        Returns:
            AudioIRPipeline instance
        """
        from config import Config
        from modules import TextChunker, TextEmbedding, VectorDatabase, RAGSystem

        # Get provider from env (default to local)
        provider = os.getenv("EMBEDDING_PROVIDER", "local")

        # Create components
        chunker = TextChunker(
            chunk_size=params.get("chunk_size", 500),
            chunk_overlap=params.get("chunk_overlap", 50),
            method=params.get("chunking_method", "recursive"),
            embedding_provider=provider
        )

        embedder = TextEmbedding(provider=provider)

        # Get dimension from embedder
        dim = embedder.embedding_dim

        vector_db = VectorDatabase(
            collection_name=f"tune_test_{int(time.time())}",
            embedding_dimension=dim
        )

        rag = RAGSystem(
            vector_db=vector_db,
            embedder=embedder,
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            temperature=params.get("llm_temperature", 0.7),
            top_k=params.get("top_k", 5)
        )

        return {
            "chunker": chunker,
            "embedder": embedder,
            "vector_db": vector_db,
            "rag": rag
        }

    def _evaluate_config(
        self,
        params: Dict,
        sample_texts: List[str]
    ) -> Dict[str, float]:
        """
        Danh gia mot cau hinh cu the

        Args:
            params: Parameter config
            sample_texts: List texts de test

        Returns:
            Dict chua metrics
        """
        from modules.evaluation_module import RAGEvaluator

        try:
            # Create pipeline
            components = self._create_pipeline(params)

            # Process sample texts
            chunks = []
            for i, text in enumerate(sample_texts):
                text_chunks = components["chunker"].chunk_text(
                    text,
                    metadata={"doc_id": f"doc_{i}"}
                )
                chunks.extend(text_chunks)

            # Encode and store
            chunks_with_emb = components["embedder"].encode_chunks(chunks)
            components["vector_db"].add_documents(chunks_with_emb)

            # Evaluate
            evaluator = RAGEvaluator(
                rag_system=components["rag"],
                embedder=components["embedder"],
                vector_db=components["vector_db"]
            )

            # Run evaluation
            results = evaluator.evaluate_end_to_end(
                self.test_data,
                k_values=[3, 5, 10]
            )

            # Cleanup
            components["vector_db"].delete_collection()

            return {
                "mrr": results["overall"]["retrieval_mrr"],
                "ndcg@5": results["overall"]["retrieval_ndcg@5"],
                "f1_score": results["overall"]["generation_f1"],
                "semantic_sim": results["overall"]["generation_semantic"],
                "latency_ms": results["overall"]["avg_latency_ms"],
                "num_chunks": len(chunks)
            }

        except Exception as e:
            print(f"Error evaluating config: {e}")
            return {
                "mrr": 0,
                "ndcg@5": 0,
                "f1_score": 0,
                "semantic_sim": 0,
                "latency_ms": 0,
                "error": str(e)
            }

    def grid_search(
        self,
        sample_texts: List[str],
        param_subset: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Thuc hien Grid Search tren cac parameters

        Args:
            sample_texts: List texts de test
            param_subset: Chi tune cac params nay (None = all)

        Returns:
            List results sorted by score
        """
        # Get params to tune
        if param_subset:
            params_to_tune = {k: v for k, v in self.param_ranges.items() if k in param_subset}
        else:
            params_to_tune = self.param_ranges

        # Generate all combinations
        param_names = list(params_to_tune.keys())
        param_values = list(params_to_tune.values())
        combinations = list(itertools.product(*param_values))

        print(f"\nGrid Search: {len(combinations)} combinations")
        print(f"Parameters: {param_names}")
        print("=" * 60)

        results = []

        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values))

            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")

            # Evaluate
            start_time = time.time()
            metrics = self._evaluate_config(params, sample_texts)
            eval_time = time.time() - start_time

            result = {
                "params": params,
                "metrics": metrics,
                "eval_time_s": eval_time,
                # Combined score (weighted)
                "score": (
                    0.3 * metrics.get("mrr", 0) +
                    0.3 * metrics.get("ndcg@5", 0) +
                    0.2 * metrics.get("f1_score", 0) +
                    0.2 * metrics.get("semantic_sim", 0)
                )
            }

            results.append(result)

            print(f"  Score: {result['score']:.4f}")
            print(f"  MRR: {metrics.get('mrr', 0):.4f}, NDCG@5: {metrics.get('ndcg@5', 0):.4f}")
            print(f"  F1: {metrics.get('f1_score', 0):.4f}, Semantic: {metrics.get('semantic_sim', 0):.4f}")

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        self.results = results
        return results

    def random_search(
        self,
        sample_texts: List[str],
        n_iterations: int = 20
    ) -> List[Dict]:
        """
        Thuc hien Random Search

        Args:
            sample_texts: List texts de test
            n_iterations: So lan thu

        Returns:
            List results sorted by score
        """
        import random

        print(f"\nRandom Search: {n_iterations} iterations")
        print("=" * 60)

        results = []

        for i in range(n_iterations):
            # Random params
            params = {
                name: random.choice(values)
                for name, values in self.param_ranges.items()
            }

            print(f"\n[{i+1}/{n_iterations}] Testing: {params}")

            # Evaluate
            start_time = time.time()
            metrics = self._evaluate_config(params, sample_texts)
            eval_time = time.time() - start_time

            result = {
                "params": params,
                "metrics": metrics,
                "eval_time_s": eval_time,
                "score": (
                    0.3 * metrics.get("mrr", 0) +
                    0.3 * metrics.get("ndcg@5", 0) +
                    0.2 * metrics.get("f1_score", 0) +
                    0.2 * metrics.get("semantic_sim", 0)
                )
            }

            results.append(result)
            print(f"  Score: {result['score']:.4f}")

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        self.results = results
        return results

    def get_best_config(self) -> Dict:
        """Lay cau hinh tot nhat"""
        if not self.results:
            raise ValueError("Chua chay tuning!")
        return self.results[0]

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Luu ket qua ra file"""
        if not filename:
            filename = f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.output_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "test_data_path": str(self.test_data_path),
            "num_test_cases": len(self.test_data),
            "param_ranges": self.param_ranges,
            "results": self.results,
            "best_config": self.results[0] if self.results else None
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\nDa luu results tai: {output_path}")
        return output_path

    def print_summary(self, top_n: int = 5):
        """In summary cua top N configs"""
        print("\n" + "=" * 60)
        print(f"TOP {top_n} CONFIGURATIONS")
        print("=" * 60)

        for i, result in enumerate(self.results[:top_n]):
            print(f"\n#{i+1} - Score: {result['score']:.4f}")
            print(f"  Params: {result['params']}")
            print(f"  Metrics:")
            for k, v in result["metrics"].items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

        if self.results:
            print("\n" + "=" * 60)
            print("BEST CONFIG:")
            print(json.dumps(self.results[0]["params"], indent=2))
            print("=" * 60)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Parameter Tuning")
    parser.add_argument(
        "--test-data",
        type=str,
        default="evaluation/datasets/test_dataset.json",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random"],
        default="random",
        help="Search method"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations (for random search)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/tuning_results",
        help="Output directory"
    )

    args = parser.parse_args()

    # Check test data exists
    if not Path(args.test_data).exists():
        print(f"Error: Test data not found at {args.test_data}")
        print("Please create test dataset first!")
        print("See: data/evaluation/test_dataset_template.json")
        return

    # Sample texts for chunking test
    sample_texts = [
        "Tri tue nhan tao (AI) la linh vuc nghien cuu ve cac he thong may tinh co kha nang thuc hien cac nhiem vu thong minh.",
        "Machine Learning la mot nhanh cua AI, cho phep may tinh hoc tu du lieu ma khong can lap trinh cu the.",
        "Deep Learning su dung cac mang neural nhieu tang de hoc cac bieu dien phuc tap tu du lieu."
    ]

    # Initialize tuner
    tuner = ParameterTuner(
        test_data_path=args.test_data,
        output_dir=args.output_dir
    )

    # Run search
    if args.method == "grid":
        # Reduced param ranges for grid search
        tuner.set_param_ranges({
            "chunk_size": [300, 500],
            "chunk_overlap": [30, 50],
            "top_k": [5, 10]
        })
        tuner.grid_search(sample_texts)
    else:
        tuner.random_search(sample_texts, n_iterations=args.iterations)

    # Print and save results
    tuner.print_summary()
    tuner.save_results()


if __name__ == "__main__":
    main()
