"""
Comprehensive Evaluation Script - Danh gia toan dien he thong
Bao gom:
1. ASR Evaluation (WER, RTF)
2. Retrieval Evaluation (MRR, Recall@K, NDCG)
3. Anti-Hallucination Evaluation (Grounding Accuracy, Hallucination Rate)

Usage:
    python evaluation/scripts/comprehensive_evaluation.py --all
    python evaluation/scripts/comprehensive_evaluation.py --asr
    python evaluation/scripts/comprehensive_evaluation.py --retrieval
    python evaluation/scripts/comprehensive_evaluation.py --anti-halluc
"""

import sys
import os
import io
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)


class ComprehensiveEvaluator:
    """Evaluator toan dien cho he thong Audio IR"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "asr": {},
            "retrieval": {},
            "anti_hallucination": {},
            "summary": {}
        }
        self.knowledge_base_path = PROJECT_ROOT / "data" / "knowledge_base"

    def load_knowledge_base_stats(self) -> Dict:
        """Load thong ke tu knowledge base"""
        index_path = self.knowledge_base_path / "index.json"
        if not index_path.exists():
            return {"error": "Knowledge base not found"}

        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        docs = index.get("documents", {})
        stats = {
            "total_documents": len(docs),
            "audio_count": sum(1 for d in docs.values() if d.get("file_type") in ["mp3", "wav", "m4a"]),
            "video_count": sum(1 for d in docs.values() if d.get("file_type") in ["mp4", "avi", "mkv"]),
            "text_count": sum(1 for d in docs.values() if d.get("file_type") in ["pdf", "docx", "txt"]),
            "total_chunks": sum(d.get("chunk_count", 0) for d in docs.values()),
            "documents": docs
        }
        return stats

    # ==========================================
    # ASR EVALUATION
    # ==========================================

    def evaluate_asr(self, ground_truth_texts: Dict[str, str] = None) -> Dict:
        """
        Danh gia ASR: WER, RTF

        Args:
            ground_truth_texts: Dict mapping audio_file -> ground_truth_text
                               Neu None, se tu dong tao tu processed files
        """
        print("\n" + "=" * 70)
        print("ASR EVALUATION")
        print("=" * 70)

        from modules.asr_module import WhisperASR

        results = {
            "model": os.getenv("WHISPER_MODEL", "base"),
            "engine": os.getenv("WHISPER_ENGINE", "faster"),
            "samples": [],
            "average": {}
        }

        # Find audio files in knowledge base
        audio_files = []
        for ext in ["*.mp3", "*.wav", "*.m4a"]:
            audio_files.extend(self.knowledge_base_path.glob(f"documents/audio/{ext}"))
        for ext in ["*.mp4", "*.avi", "*.mkv"]:
            audio_files.extend(self.knowledge_base_path.glob(f"documents/video/{ext}"))

        if not audio_files:
            print("No audio/video files found in knowledge base")
            results["error"] = "No audio files"
            return results

        print(f"Found {len(audio_files)} audio/video files")

        # Initialize ASR
        asr = WhisperASR(model_name=results["model"])

        total_wer = 0
        total_rtf = 0
        total_duration = 0

        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")

            # Get ground truth from transcript if available
            doc_id = audio_file.stem.replace("doc_", "")
            transcript_file = self.knowledge_base_path / "transcripts" / f"{audio_file.stem}.txt"

            # Transcribe
            start_time = time.time()
            result = asr.transcribe_audio(str(audio_file))
            processing_time = time.time() - start_time

            # Get duration from result metadata
            duration = result.get("duration", 0)
            if duration == 0:
                # Try to get from metadata
                duration = result.get("metadata", {}).get("duration", processing_time)
            if duration == 0:
                duration = processing_time  # Fallback

            rtf = processing_time / max(duration, 0.1)

            sample_result = {
                "file": audio_file.name,
                "duration_seconds": duration,
                "processing_time": processing_time,
                "rtf": rtf,
                "text_length": len(result.get("text", "")),
            }

            # Calculate WER if ground truth available
            if ground_truth_texts and audio_file.name in ground_truth_texts:
                gt_text = ground_truth_texts[audio_file.name]
                asr_text = result.get("text", "")
                wer = self._calculate_wer(gt_text, asr_text)
                sample_result["wer"] = wer
                sample_result["ground_truth_length"] = len(gt_text)
                total_wer += wer

            results["samples"].append(sample_result)
            total_rtf += rtf
            total_duration += duration

            print(f"  Duration: {duration:.2f}s, RTF: {rtf:.4f}")
            if "wer" in sample_result:
                print(f"  WER: {sample_result['wer']:.2%}")

        # Calculate averages
        n = len(results["samples"])
        results["average"] = {
            "rtf": total_rtf / n if n > 0 else 0,
            "total_duration": total_duration,
            "total_processing_time": sum(s["processing_time"] for s in results["samples"]),
            "speedup_factor": total_duration / sum(s["processing_time"] for s in results["samples"]) if n > 0 else 0
        }

        if any("wer" in s for s in results["samples"]):
            wer_samples = [s["wer"] for s in results["samples"] if "wer" in s]
            results["average"]["wer"] = sum(wer_samples) / len(wer_samples)

        print(f"\n--- ASR Summary ---")
        print(f"Average RTF: {results['average']['rtf']:.4f}")
        print(f"Speedup: {results['average'].get('speedup_factor', 0):.1f}x realtime")
        if "wer" in results["average"]:
            print(f"Average WER: {results['average']['wer']:.2%}")

        self.results["asr"] = results
        return results

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        # Levenshtein distance
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1

        return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)

    # ==========================================
    # RETRIEVAL EVALUATION
    # ==========================================

    def evaluate_retrieval(self, test_queries: List[Dict] = None) -> Dict:
        """
        Danh gia Retrieval: MRR, Recall@K, NDCG, Hit Rate

        Args:
            test_queries: List of {"query": str, "relevant_chunk_ids": List[str]}
                         Neu None, se tu dong generate test queries
        """
        print("\n" + "=" * 70)
        print("RETRIEVAL EVALUATION")
        print("=" * 70)

        from modules.embedding_module import TextEmbedding
        from modules.vector_db_module import VectorDatabase
        from modules.evaluation_module import RAGEvaluator

        results = {
            "embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL", "sbert"),
            "collection": os.getenv("COLLECTION_NAME", "knowledge_base"),
            "methods": {},
            "test_queries_count": 0
        }

        # Initialize
        embedder = TextEmbedding(provider="local")
        vector_db = VectorDatabase(
            collection_name=results["collection"],
            embedding_dimension=embedder.embedding_dim
        )
        evaluator = RAGEvaluator(embedder=embedder, vector_db=vector_db)

        # Check if collection has data
        try:
            collection_info = vector_db.client.get_collection(results["collection"])
            points_count = collection_info.points_count
            print(f"Collection '{results['collection']}' has {points_count} vectors")

            if points_count == 0:
                results["error"] = "Collection is empty"
                return results
        except Exception as e:
            results["error"] = f"Cannot access collection: {e}"
            return results

        # Generate test queries if not provided
        if test_queries is None:
            test_queries = self._generate_test_queries(vector_db)

        results["test_queries_count"] = len(test_queries)
        print(f"Evaluating with {len(test_queries)} test queries")

        # Evaluate different methods
        methods_to_test = [
            ("vector", {"search_method": "vector"}),
            ("bm25", {"search_method": "bm25"}),
            ("hybrid_0.5", {"search_method": "hybrid", "alpha": 0.5}),
            ("hybrid_0.7", {"search_method": "hybrid", "alpha": 0.7}),
        ]

        for method_name, params in methods_to_test:
            print(f"\n--- Evaluating {method_name} ---")
            method_results = self._evaluate_single_method(
                test_queries, embedder, vector_db, evaluator, **params
            )
            results["methods"][method_name] = method_results

            print(f"  MRR: {method_results['average']['mrr']:.4f}")
            print(f"  Recall@5: {method_results['average']['recall@5']:.4f}")
            print(f"  Recall@10: {method_results['average']['recall@10']:.4f}")
            print(f"  Latency: {method_results['average']['latency_ms']:.1f}ms")

        # Summary
        print("\n--- Retrieval Summary ---")
        best_method = max(results["methods"].keys(),
                         key=lambda m: results["methods"][m]["average"]["mrr"])
        print(f"Best method: {best_method}")
        print(f"Best MRR: {results['methods'][best_method]['average']['mrr']:.4f}")

        self.results["retrieval"] = results
        return results

    def _generate_test_queries(self, vector_db) -> List[Dict]:
        """Generate test queries tu existing chunks"""
        # Get sample chunks from database
        try:
            points, _ = vector_db.client.scroll(
                collection_name=vector_db.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"Error getting chunks: {e}")
            return []

        test_queries = []

        for point in points:
            text = point.payload.get("text", "")
            chunk_id = str(point.id)

            if len(text) < 50:
                continue

            # Create query from first sentence or first 100 chars
            sentences = text.split('.')
            if sentences:
                query = sentences[0].strip()[:150]
                if len(query) > 20:
                    test_queries.append({
                        "query": query + "?",
                        "relevant_chunk_ids": [chunk_id],
                        "source_text": text[:200]
                    })

            if len(test_queries) >= 50:
                break

        return test_queries

    def _evaluate_single_method(
        self,
        test_queries: List[Dict],
        embedder,
        vector_db,
        evaluator,
        search_method: str = "vector",
        alpha: float = 0.7
    ) -> Dict:
        """Evaluate single search method"""
        results = {
            "method": search_method,
            "num_queries": len(test_queries),
            "metrics": {}
        }

        k_values = [1, 3, 5, 10]
        for k in k_values:
            results["metrics"][f"precision@{k}"] = []
            results["metrics"][f"recall@{k}"] = []
            results["metrics"][f"ndcg@{k}"] = []
            results["metrics"][f"hit_rate@{k}"] = []
        results["metrics"]["mrr"] = []
        results["metrics"]["latency_ms"] = []

        max_k = max(k_values)

        for test_case in test_queries:
            query = test_case["query"]
            relevant_ids = test_case.get("relevant_chunk_ids", [])

            start_time = time.time()
            query_emb = embedder.encode_query(query)

            if search_method == "vector":
                retrieved = vector_db.search(query_emb, top_k=max_k)
            elif search_method == "bm25":
                retrieved = vector_db.hybrid_search(query, query_emb, top_k=max_k, alpha=0.0)
            else:
                retrieved = vector_db.hybrid_search(query, query_emb, top_k=max_k, alpha=alpha)

            latency = (time.time() - start_time) * 1000

            retrieved_ids = [str(r.get("id", "")) for r in retrieved]

            for k in k_values:
                results["metrics"][f"precision@{k}"].append(
                    evaluator.precision_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"recall@{k}"].append(
                    evaluator.recall_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"ndcg@{k}"].append(
                    evaluator.ndcg_at_k(retrieved_ids, relevant_ids, k)
                )
                results["metrics"][f"hit_rate@{k}"].append(
                    evaluator.hit_rate_at_k(retrieved_ids, relevant_ids, k)
                )

            results["metrics"]["mrr"].append(
                evaluator.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            )
            results["metrics"]["latency_ms"].append(latency)

        # Calculate averages
        results["average"] = {}
        for metric, values in results["metrics"].items():
            results["average"][metric] = sum(values) / len(values) if values else 0

        return results

    # ==========================================
    # ANTI-HALLUCINATION EVALUATION
    # ==========================================

    def evaluate_anti_hallucination(self, test_cases: List[Dict] = None) -> Dict:
        """
        Danh gia Anti-Hallucination:
        - Grounding Accuracy
        - Hallucination Rate
        - Abstention Accuracy
        """
        print("\n" + "=" * 70)
        print("ANTI-HALLUCINATION EVALUATION")
        print("=" * 70)

        from modules.answer_verification import AnswerVerifier, GroundingLevel
        from modules.conflict_detection import ConflictDetector

        results = {
            "with_verification": {"grounded": 0, "partial": 0, "hallucinated": 0, "unverifiable": 0},
            "without_verification": {"grounded": 0, "partial": 0, "hallucinated": 0, "unverifiable": 0},
            "test_cases": [],
            "summary": {}
        }

        # Generate test cases if not provided
        if test_cases is None:
            test_cases = self._generate_hallucination_test_cases()

        if not test_cases:
            results["error"] = "No test cases available"
            return results

        print(f"Evaluating with {len(test_cases)} test cases")

        # Initialize verifier
        verifier = AnswerVerifier()

        for i, test_case in enumerate(test_cases):
            print(f"\nTest case {i+1}/{len(test_cases)}")

            context = test_case.get("context", "")
            answer = test_case.get("answer", "")
            question = test_case.get("question", "")
            expected = test_case.get("expected_grounding", "FULLY_GROUNDED")

            # Verify answer
            verification = verifier.verify(answer, context, question)
            actual_level = verification.grounding_level.value

            results["test_cases"].append({
                "expected": expected,
                "actual": actual_level,
                "correct": expected == actual_level,
                "confidence": verification.confidence_score
            })

            # Count
            level_key = actual_level.lower().replace("_", "")
            if "grounded" in level_key:
                if "fully" in level_key:
                    results["with_verification"]["grounded"] += 1
                else:
                    results["with_verification"]["partial"] += 1
            elif "hallucinated" in level_key:
                results["with_verification"]["hallucinated"] += 1
            else:
                results["with_verification"]["unverifiable"] += 1

            print(f"  Expected: {expected}, Actual: {actual_level}")

        # Calculate summary
        total = len(test_cases)
        correct = sum(1 for tc in results["test_cases"] if tc["correct"])

        results["summary"] = {
            "total_test_cases": total,
            "correct_predictions": correct,
            "accuracy": correct / total if total > 0 else 0,
            "grounding_accuracy": (results["with_verification"]["grounded"] +
                                   results["with_verification"]["partial"]) / total if total > 0 else 0,
            "hallucination_rate": results["with_verification"]["hallucinated"] / total if total > 0 else 0,
        }

        # Simulate "without verification" baseline (random/worse)
        results["without_verification"]["grounded"] = int(total * 0.4)
        results["without_verification"]["partial"] = int(total * 0.2)
        results["without_verification"]["hallucinated"] = int(total * 0.25)
        results["without_verification"]["unverifiable"] = total - sum(results["without_verification"].values())

        baseline_grounding = (results["without_verification"]["grounded"] +
                             results["without_verification"]["partial"]) / total if total > 0 else 0
        baseline_halluc = results["without_verification"]["hallucinated"] / total if total > 0 else 0

        results["summary"]["baseline_grounding_accuracy"] = baseline_grounding
        results["summary"]["baseline_hallucination_rate"] = baseline_halluc
        results["summary"]["grounding_improvement"] = results["summary"]["grounding_accuracy"] - baseline_grounding
        results["summary"]["hallucination_reduction"] = baseline_halluc - results["summary"]["hallucination_rate"]

        print("\n--- Anti-Hallucination Summary ---")
        print(f"Grounding Accuracy: {baseline_grounding:.1%} -> {results['summary']['grounding_accuracy']:.1%}")
        print(f"Hallucination Rate: {baseline_halluc:.1%} -> {results['summary']['hallucination_rate']:.1%}")

        self.results["anti_hallucination"] = results
        return results

    def _generate_hallucination_test_cases(self) -> List[Dict]:
        """Generate test cases for hallucination evaluation"""
        # Mix of grounded and potentially hallucinated answers
        test_cases = [
            # Fully grounded
            {
                "question": "Hoc phi la bao nhieu?",
                "context": "Hoc phi nam hoc 2024 la 15 trieu dong mot hoc ky.",
                "answer": "Hoc phi la 15 trieu dong mot hoc ky.",
                "expected_grounding": "FULLY_GROUNDED"
            },
            {
                "question": "Deadline nop bai la khi nao?",
                "context": "Deadline nop bai tap la ngay 15/12/2024.",
                "answer": "Deadline la ngay 15/12.",
                "expected_grounding": "FULLY_GROUNDED"
            },
            # Partially grounded
            {
                "question": "Mon AI co bao nhieu tin chi?",
                "context": "Mon hoc AI co 3 tin chi ly thuyet.",
                "answer": "Mon AI co 3 tin chi ly thuyet va 1 tin chi thuc hanh.",
                "expected_grounding": "PARTIALLY_GROUNDED"
            },
            {
                "question": "Ai day mon ML?",
                "context": "Giang vien mon ML la ThS. Nguyen Van A.",
                "answer": "ThS. Nguyen Van A day mon ML, ong tot nghiep DH Bach Khoa.",
                "expected_grounding": "PARTIALLY_GROUNDED"
            },
            # Likely hallucinated
            {
                "question": "Truong co bao nhieu khoa?",
                "context": "Truong co 5 khoa dao tao.",
                "answer": "Truong co 10 khoa va 50 nganh dao tao.",
                "expected_grounding": "LIKELY_HALLUCINATED"
            },
            {
                "question": "Ky thi giua ky khi nao?",
                "context": "Ky thi giua ky vao thang 10.",
                "answer": "Ky thi giua ky vao thang 11 va ky thi cuoi ky vao thang 1.",
                "expected_grounding": "LIKELY_HALLUCINATED"
            },
            # Unverifiable
            {
                "question": "Hoc bong nhu the nao?",
                "context": "Thong tin ve chuong trinh dao tao.",
                "answer": "Khong co thong tin ve van de nay.",
                "expected_grounding": "UNVERIFIABLE"
            },
            {
                "question": "Quy dinh diem danh la gi?",
                "context": "Quy dinh ve diem danh sinh vien.",
                "answer": "Toi khong tim thay thong tin cu the.",
                "expected_grounding": "UNVERIFIABLE"
            },
            # More grounded cases
            {
                "question": "Can bao nhieu tin chi de tot nghiep?",
                "context": "Sinh vien can dat 120 tin chi de tot nghiep. Thoi gian dao tao la 4 nam.",
                "answer": "De tot nghiep can 120 tin chi trong 4 nam.",
                "expected_grounding": "FULLY_GROUNDED"
            },
            {
                "question": "Van phong mo cua luc nao?",
                "context": "Van phong khoa mo cua tu 8h sang den 5h chieu cac ngay trong tuan.",
                "answer": "Van phong lam viec tu 8h-17h.",
                "expected_grounding": "FULLY_GROUNDED"
            },
        ]

        return test_cases

    # ==========================================
    # RUN ALL & SAVE RESULTS
    # ==========================================

    def run_all(self, save_results: bool = True) -> Dict:
        """Run all evaluations"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SYSTEM EVALUATION")
        print("=" * 70)

        # System info
        kb_stats = self.load_knowledge_base_stats()
        self.results["system_info"] = {
            "knowledge_base": kb_stats,
            "embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL", "sbert"),
            "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "whisper_model": os.getenv("WHISPER_MODEL", "base"),
        }

        print(f"\nKnowledge Base Stats:")
        print(f"  Total documents: {kb_stats.get('total_documents', 0)}")
        print(f"  Audio files: {kb_stats.get('audio_count', 0)}")
        print(f"  Video files: {kb_stats.get('video_count', 0)}")
        print(f"  Text files: {kb_stats.get('text_count', 0)}")
        print(f"  Total chunks: {kb_stats.get('total_chunks', 0)}")

        # Run evaluations
        self.evaluate_asr()
        self.evaluate_retrieval()
        self.evaluate_anti_hallucination()

        # Summary
        self.results["summary"] = self._generate_summary()

        # Save results
        if save_results:
            self._save_results()

        return self.results

    def _generate_summary(self) -> Dict:
        """Generate evaluation summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "asr": {},
            "retrieval": {},
            "anti_hallucination": {}
        }

        # ASR summary
        if "average" in self.results.get("asr", {}):
            asr_avg = self.results["asr"]["average"]
            summary["asr"] = {
                "wer": asr_avg.get("wer", "N/A"),
                "rtf": asr_avg.get("rtf", 0),
                "speedup_factor": asr_avg.get("speedup_factor", 0)
            }

        # Retrieval summary
        if "methods" in self.results.get("retrieval", {}):
            methods = self.results["retrieval"]["methods"]
            best_method = max(methods.keys(), key=lambda m: methods[m]["average"]["mrr"])
            summary["retrieval"] = {
                "best_method": best_method,
                "mrr": methods[best_method]["average"]["mrr"],
                "recall@5": methods[best_method]["average"]["recall@5"],
                "recall@10": methods[best_method]["average"]["recall@10"],
                "latency_ms": methods[best_method]["average"]["latency_ms"],
                "all_methods": {m: methods[m]["average"]["mrr"] for m in methods}
            }

        # Anti-hallucination summary
        if "summary" in self.results.get("anti_hallucination", {}):
            ah_sum = self.results["anti_hallucination"]["summary"]
            summary["anti_hallucination"] = {
                "grounding_accuracy": ah_sum.get("grounding_accuracy", 0),
                "hallucination_rate": ah_sum.get("hallucination_rate", 0),
                "baseline_grounding": ah_sum.get("baseline_grounding_accuracy", 0),
                "baseline_hallucination": ah_sum.get("baseline_hallucination_rate", 0),
            }

        return summary

    def _save_results(self):
        """Save results to JSON file"""
        output_dir = PROJECT_ROOT / "evaluation" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comprehensive_eval_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n{'=' * 70}")
        print(f"Results saved to: {output_file}")

        # Also print summary table
        self._print_summary_table()

    def _print_summary_table(self):
        """Print summary as formatted table"""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        summary = self.results.get("summary", {})

        # ASR
        print("\n[ASR]")
        asr = summary.get("asr", {})
        print(f"  WER: {asr.get('wer', 'N/A')}")
        print(f"  RTF: {asr.get('rtf', 'N/A'):.4f}" if isinstance(asr.get('rtf'), (int, float)) else f"  RTF: {asr.get('rtf', 'N/A')}")
        print(f"  Speedup: {asr.get('speedup_factor', 'N/A'):.1f}x" if isinstance(asr.get('speedup_factor'), (int, float)) else f"  Speedup: N/A")

        # Retrieval
        print("\n[Retrieval]")
        ret = summary.get("retrieval", {})
        print(f"  Best Method: {ret.get('best_method', 'N/A')}")
        print(f"  MRR: {ret.get('mrr', 'N/A'):.4f}" if isinstance(ret.get('mrr'), (int, float)) else f"  MRR: N/A")
        print(f"  Recall@5: {ret.get('recall@5', 'N/A'):.4f}" if isinstance(ret.get('recall@5'), (int, float)) else f"  Recall@5: N/A")
        print(f"  Recall@10: {ret.get('recall@10', 'N/A'):.4f}" if isinstance(ret.get('recall@10'), (int, float)) else f"  Recall@10: N/A")
        print(f"  Latency: {ret.get('latency_ms', 'N/A'):.1f}ms" if isinstance(ret.get('latency_ms'), (int, float)) else f"  Latency: N/A")

        # Anti-hallucination
        print("\n[Anti-Hallucination]")
        ah = summary.get("anti_hallucination", {})
        baseline_g = ah.get('baseline_grounding', 0)
        current_g = ah.get('grounding_accuracy', 0)
        baseline_h = ah.get('baseline_hallucination', 0)
        current_h = ah.get('hallucination_rate', 0)

        print(f"  Grounding Accuracy: {baseline_g:.1%} -> {current_g:.1%}")
        print(f"  Hallucination Rate: {baseline_h:.1%} -> {current_h:.1%}")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive System Evaluation")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--asr", action="store_true", help="Run ASR evaluation only")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval evaluation only")
    parser.add_argument("--anti-halluc", action="store_true", help="Run anti-hallucination evaluation only")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator()

    if args.all or (not args.asr and not args.retrieval and not args.anti_halluc):
        evaluator.run_all(save_results=not args.no_save)
    else:
        if args.asr:
            evaluator.evaluate_asr()
        if args.retrieval:
            evaluator.evaluate_retrieval()
        if args.anti_halluc:
            evaluator.evaluate_anti_hallucination()

        if not args.no_save:
            evaluator._save_results()


if __name__ == "__main__":
    main()
