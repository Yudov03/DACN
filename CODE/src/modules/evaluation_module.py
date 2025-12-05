"""
Evaluation Module - Do luong hieu suat cua he thong RAG
Bao gom: Retrieval metrics, Generation metrics, Latency
"""

import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class RAGEvaluator:
    """
    Lop danh gia hieu suat he thong RAG

    Metrics:
    - Retrieval: Precision@K, Recall@K, MRR, NDCG, Hit Rate
    - Generation: Exact Match, F1 Score, BLEU, Semantic Similarity
    - Latency: Query time, Embedding time, Total time
    """

    def __init__(
        self,
        rag_system=None,
        embedder=None,
        vector_db=None
    ):
        """
        Khoi tao evaluator

        Args:
            rag_system: RAGSystem instance
            embedder: TextEmbedding instance
            vector_db: VectorDatabase instance
        """
        self.rag_system = rag_system
        self.embedder = embedder
        self.vector_db = vector_db
        self.results = []

    # ==========================================
    # RETRIEVAL METRICS
    # ==========================================

    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Tinh Precision@K

        Args:
            retrieved_ids: List IDs duoc retrieve
            relevant_ids: List IDs thuc su relevant
            k: So luong top results

        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0

        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)

        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)

        return relevant_retrieved / k

    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Tinh Recall@K

        Args:
            retrieved_ids: List IDs duoc retrieve
            relevant_ids: List IDs thuc su relevant
            k: So luong top results

        Returns:
            Recall@K score
        """
        if len(relevant_ids) == 0:
            return 0.0

        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        relevant_retrieved = len(retrieved_k & relevant_set)

        return relevant_retrieved / len(relevant_set)

    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Tinh Mean Reciprocal Rank (MRR)

        Args:
            retrieved_ids: List IDs duoc retrieve
            relevant_ids: List IDs thuc su relevant

        Returns:
            MRR score
        """
        relevant_set = set(relevant_ids)

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Tinh Normalized Discounted Cumulative Gain (NDCG@K)

        Args:
            retrieved_ids: List IDs duoc retrieve
            relevant_ids: List IDs thuc su relevant
            k: So luong top results
            relevance_scores: Dict {doc_id: relevance_score} (optional)

        Returns:
            NDCG@K score
        """
        if relevance_scores is None:
            # Binary relevance
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 vi rank bat dau tu 1

        # Ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def hit_rate_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Tinh Hit Rate@K (co it nhat 1 relevant trong top-k)

        Args:
            retrieved_ids: List IDs duoc retrieve
            relevant_ids: List IDs thuc su relevant
            k: So luong top results

        Returns:
            1.0 neu hit, 0.0 neu miss
        """
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        return 1.0 if len(retrieved_k & relevant_set) > 0 else 0.0

    # ==========================================
    # GENERATION METRICS
    # ==========================================

    def exact_match(
        self,
        prediction: str,
        ground_truth: str,
        normalize: bool = True
    ) -> float:
        """
        Tinh Exact Match score

        Args:
            prediction: Cau tra loi du doan
            ground_truth: Cau tra loi dung
            normalize: Co normalize text khong

        Returns:
            1.0 neu match, 0.0 neu khong
        """
        if normalize:
            pred = self._normalize_text(prediction)
            truth = self._normalize_text(ground_truth)
        else:
            pred = prediction
            truth = ground_truth

        return 1.0 if pred == truth else 0.0

    def f1_score(
        self,
        prediction: str,
        ground_truth: str
    ) -> float:
        """
        Tinh F1 Score (token-level)

        Args:
            prediction: Cau tra loi du doan
            ground_truth: Cau tra loi dung

        Returns:
            F1 score
        """
        pred_tokens = set(self._normalize_text(prediction).split())
        truth_tokens = set(self._normalize_text(ground_truth).split())

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0

        common = pred_tokens & truth_tokens

        if len(common) == 0:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)

        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def bleu_score(
        self,
        prediction: str,
        ground_truth: str,
        n_gram: int = 4
    ) -> float:
        """
        Tinh BLEU Score (simplified)

        Args:
            prediction: Cau tra loi du doan
            ground_truth: Cau tra loi dung
            n_gram: Max n-gram

        Returns:
            BLEU score
        """
        pred_tokens = self._normalize_text(prediction).split()
        truth_tokens = self._normalize_text(ground_truth).split()

        if len(pred_tokens) == 0:
            return 0.0

        # N-gram precision
        precisions = []
        for n in range(1, min(n_gram + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            truth_ngrams = self._get_ngrams(truth_tokens, n)

            if len(pred_ngrams) == 0:
                continue

            matches = sum(1 for ng in pred_ngrams if ng in truth_ngrams)
            precisions.append(matches / len(pred_ngrams))

        if len(precisions) == 0:
            return 0.0

        # Geometric mean
        log_precisions = [np.log(p) if p > 0 else -np.inf for p in precisions]
        avg_log = np.mean(log_precisions)

        if avg_log == -np.inf:
            return 0.0

        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(truth_tokens) / max(len(pred_tokens), 1)))

        return bp * np.exp(avg_log)

    def semantic_similarity(
        self,
        prediction: str,
        ground_truth: str
    ) -> float:
        """
        Tinh Semantic Similarity su dung embeddings

        Args:
            prediction: Cau tra loi du doan
            ground_truth: Cau tra loi dung

        Returns:
            Cosine similarity score
        """
        if self.embedder is None:
            raise ValueError("Embedder chua duoc khoi tao!")

        pred_emb = self.embedder.encode_query(prediction)
        truth_emb = self.embedder.encode_query(ground_truth)

        return self.embedder.compute_similarity(pred_emb, truth_emb)

    # ==========================================
    # EVALUATION PIPELINE
    # ==========================================

    def evaluate_retrieval(
        self,
        test_data: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Danh gia retrieval tren test dataset

        Args:
            test_data: List cac test cases
                [{"query": "...", "relevant_doc_ids": ["id1", "id2"]}]
            k_values: List cac gia tri k de danh gia

        Returns:
            Dict chua metrics
        """
        if self.embedder is None or self.vector_db is None:
            raise ValueError("Embedder va VectorDB chua duoc khoi tao!")

        results = {
            "num_queries": len(test_data),
            "k_values": k_values,
            "metrics": {}
        }

        # Initialize metrics
        for k in k_values:
            results["metrics"][f"precision@{k}"] = []
            results["metrics"][f"recall@{k}"] = []
            results["metrics"][f"ndcg@{k}"] = []
            results["metrics"][f"hit_rate@{k}"] = []
        results["metrics"]["mrr"] = []
        results["metrics"]["latency_ms"] = []

        # Evaluate each query
        for test_case in test_data:
            query = test_case["query"]
            relevant_ids = test_case.get("relevant_doc_ids", [])

            # Measure retrieval time
            start_time = time.time()
            query_embedding = self.embedder.encode_query(query)
            retrieved = self.vector_db.search(query_embedding, top_k=max(k_values))
            latency = (time.time() - start_time) * 1000

            retrieved_ids = [doc["id"] for doc in retrieved]

            # Calculate metrics
            for k in k_values:
                results["metrics"][f"precision@{k}"].append(
                    self.precision_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"recall@{k}"].append(
                    self.recall_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"ndcg@{k}"].append(
                    self.ndcg_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"hit_rate@{k}"].append(
                    self.hit_rate_at_k(retrieved_ids, relevant_ids, k)
                )

            results["metrics"]["mrr"].append(
                self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            )
            results["metrics"]["latency_ms"].append(latency)

        # Average metrics
        results["average"] = {}
        for metric_name, values in results["metrics"].items():
            results["average"][metric_name] = np.mean(values)

        return results

    def evaluate_generation(
        self,
        test_data: List[Dict],
        use_semantic: bool = True
    ) -> Dict[str, Any]:
        """
        Danh gia generation (answer quality) tren test dataset

        Args:
            test_data: List cac test cases
                [{"query": "...", "ground_truth_answer": "..."}]
            use_semantic: Co tinh semantic similarity khong

        Returns:
            Dict chua metrics
        """
        if self.rag_system is None:
            raise ValueError("RAG System chua duoc khoi tao!")

        results = {
            "num_queries": len(test_data),
            "metrics": {
                "exact_match": [],
                "f1_score": [],
                "bleu_score": [],
                "latency_ms": []
            }
        }

        if use_semantic and self.embedder is not None:
            results["metrics"]["semantic_similarity"] = []

        # Evaluate each query
        for test_case in test_data:
            query = test_case["query"]
            ground_truth = test_case.get("ground_truth_answer", "")

            # Get prediction
            start_time = time.time()
            response = self.rag_system.query(query)
            latency = (time.time() - start_time) * 1000

            prediction = response.get("answer", "")

            # Calculate metrics
            results["metrics"]["exact_match"].append(
                self.exact_match(prediction, ground_truth)
            )
            results["metrics"]["f1_score"].append(
                self.f1_score(prediction, ground_truth)
            )
            results["metrics"]["bleu_score"].append(
                self.bleu_score(prediction, ground_truth)
            )
            results["metrics"]["latency_ms"].append(latency)

            if use_semantic and self.embedder is not None:
                results["metrics"]["semantic_similarity"].append(
                    self.semantic_similarity(prediction, ground_truth)
                )

        # Average metrics
        results["average"] = {}
        for metric_name, values in results["metrics"].items():
            results["average"][metric_name] = np.mean(values)

        return results

    def evaluate_end_to_end(
        self,
        test_data: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Danh gia toan bo pipeline (retrieval + generation)

        Args:
            test_data: List cac test cases day du
                [{
                    "query": "...",
                    "relevant_doc_ids": ["id1", "id2"],
                    "ground_truth_answer": "..."
                }]
            k_values: List cac gia tri k cho retrieval

        Returns:
            Dict chua tat ca metrics
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(test_data),
            "retrieval": None,
            "generation": None,
            "overall": {}
        }

        # Retrieval evaluation
        retrieval_data = [
            {"query": t["query"], "relevant_doc_ids": t.get("relevant_doc_ids", [])}
            for t in test_data
        ]
        results["retrieval"] = self.evaluate_retrieval(retrieval_data, k_values)

        # Generation evaluation
        generation_data = [
            {"query": t["query"], "ground_truth_answer": t.get("ground_truth_answer", "")}
            for t in test_data
        ]
        results["generation"] = self.evaluate_generation(generation_data)

        # Overall summary
        results["overall"] = {
            "retrieval_mrr": results["retrieval"]["average"]["mrr"],
            "retrieval_ndcg@5": results["retrieval"]["average"].get("ndcg@5", 0),
            "generation_f1": results["generation"]["average"]["f1_score"],
            "generation_semantic": results["generation"]["average"].get("semantic_similarity", 0),
            "avg_latency_ms": results["generation"]["average"]["latency_ms"]
        }

        return results

    # ==========================================
    # HELPER METHODS
    # ==========================================

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower().strip()
        # Remove punctuation
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from token list"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def save_results(
        self,
        results: Dict,
        output_path: str
    ) -> Path:
        """Luu results ra file JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"Da luu results tai: {output_path}")
        return output_path

    def print_results(self, results: Dict):
        """In results ra console"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        if "retrieval" in results and results["retrieval"]:
            print("\n--- RETRIEVAL METRICS ---")
            for metric, value in results["retrieval"]["average"].items():
                print(f"  {metric}: {value:.4f}")

        if "generation" in results and results["generation"]:
            print("\n--- GENERATION METRICS ---")
            for metric, value in results["generation"]["average"].items():
                print(f"  {metric}: {value:.4f}")

        if "overall" in results:
            print("\n--- OVERALL SUMMARY ---")
            for metric, value in results["overall"].items():
                print(f"  {metric}: {value:.4f}")

        print("=" * 60)


# Test function
if __name__ == "__main__":
    print("Evaluation Module initialized successfully!")

    # Test metrics calculation
    evaluator = RAGEvaluator()

    # Test retrieval metrics
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc6"]

    print(f"\nRetrieval metrics test:")
    print(f"  Precision@3: {evaluator.precision_at_k(retrieved, relevant, 3):.4f}")
    print(f"  Recall@3: {evaluator.recall_at_k(retrieved, relevant, 3):.4f}")
    print(f"  MRR: {evaluator.mean_reciprocal_rank(retrieved, relevant):.4f}")
    print(f"  NDCG@5: {evaluator.ndcg_at_k(retrieved, relevant, 5):.4f}")
    print(f"  Hit Rate@3: {evaluator.hit_rate_at_k(retrieved, relevant, 3):.4f}")

    # Test generation metrics
    prediction = "Machine Learning la mot nhanh cua AI"
    ground_truth = "Machine Learning la linh vuc cua tri tue nhan tao"

    print(f"\nGeneration metrics test:")
    print(f"  F1 Score: {evaluator.f1_score(prediction, ground_truth):.4f}")
    print(f"  BLEU Score: {evaluator.bleu_score(prediction, ground_truth):.4f}")
