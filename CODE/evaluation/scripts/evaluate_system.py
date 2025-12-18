"""
Script danh gia hieu suat he thong Audio IR
Bao gom: Retrieval metrics, Generation metrics
"""

import sys
import os
import io
from pathlib import Path
import json
import time

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


# Sample Vietnamese QA dataset for evaluation
SAMPLE_EVAL_DATA = [
    {
        "id": "q1",
        "query": "Machine learning la gi?",
        "context": "Machine learning la mot nhanh cua tri tue nhan tao, cho phep may tinh tu hoc tu du lieu ma khong can lap trinh cu the.",
        "ground_truth_answer": "Machine learning la mot nhanh cua tri tue nhan tao, cho phep may tinh tu hoc tu du lieu.",
        "relevant_chunks": ["chunk_ml_1", "chunk_ml_2"]
    },
    {
        "id": "q2",
        "query": "Deep learning khac gi voi machine learning?",
        "context": "Deep learning la mot phan cua machine learning, su dung mang neural nhieu lop de xu ly du lieu phuc tap.",
        "ground_truth_answer": "Deep learning su dung mang neural nhieu lop, la mot phan cua machine learning.",
        "relevant_chunks": ["chunk_dl_1"]
    },
    {
        "id": "q3",
        "query": "NLP dung de lam gi?",
        "context": "Natural Language Processing giup may tinh hieu va xu ly ngon ngu tu nhien cua con nguoi, bao gom dich thuat, phan tich cam xuc.",
        "ground_truth_answer": "NLP giup may tinh hieu va xu ly ngon ngu tu nhien.",
        "relevant_chunks": ["chunk_nlp_1", "chunk_nlp_2"]
    },
    {
        "id": "q4",
        "query": "Whisper la gi?",
        "context": "Whisper la mo hinh ASR cua OpenAI, co the chuyen doi giong noi thanh van ban voi do chinh xac cao, ho tro nhieu ngon ngu.",
        "ground_truth_answer": "Whisper la mo hinh ASR cua OpenAI de chuyen giong noi thanh van ban.",
        "relevant_chunks": ["chunk_whisper_1"]
    },
    {
        "id": "q5",
        "query": "Vector database hoat dong nhu the nao?",
        "context": "Vector database luu tru cac embedding vectors va su dung thuat toan nhu HNSW de tim kiem tuong tu nhanh chong.",
        "ground_truth_answer": "Vector database luu tru embedding vectors va tim kiem tuong tu.",
        "relevant_chunks": ["chunk_vectordb_1"]
    },
    {
        "id": "q6",
        "query": "RAG la gi va tai sao quan trong?",
        "context": "RAG ket hop retrieval va generation de tao cau tra loi dua tren nguon du lieu cu the, giam hallucination cua LLM.",
        "ground_truth_answer": "RAG ket hop retrieval va generation de giam hallucination.",
        "relevant_chunks": ["chunk_rag_1", "chunk_rag_2"]
    },
    {
        "id": "q7",
        "query": "Embedding la gi?",
        "context": "Embedding la cach bieu dien van ban hoac du lieu duoi dang vector so, giup may tinh hieu y nghia ngu nghia.",
        "ground_truth_answer": "Embedding la bieu dien van ban duoi dang vector.",
        "relevant_chunks": ["chunk_emb_1"]
    },
    {
        "id": "q8",
        "query": "Transformer architecture la gi?",
        "context": "Transformer la kien truc neural network su dung attention mechanism, la nen tang cua cac mo hinh nhu BERT, GPT.",
        "ground_truth_answer": "Transformer su dung attention mechanism, la nen tang cua BERT va GPT.",
        "relevant_chunks": ["chunk_transformer_1"]
    },
]


def create_test_chunks():
    """Tao test chunks tu sample data"""
    chunks = []
    for i, item in enumerate(SAMPLE_EVAL_DATA):
        chunk = {
            "text": item["context"],
            "chunk_id": f"chunk_{i}",
            "audio_file": f"test_audio_{i}.mp3",
            "start_time": float(i * 10),
            "end_time": float(i * 10 + 10),
            "query_id": item["id"]
        }
        chunks.append(chunk)
    return chunks


def evaluate_retrieval_only(embedder, vector_db, test_data, k_values=[1, 3, 5]):
    """Danh gia chi rieng retrieval"""
    from modules.evaluation_module import RAGEvaluator

    evaluator = RAGEvaluator(embedder=embedder, vector_db=vector_db)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    results = {
        "precision": {k: [] for k in k_values},
        "recall": {k: [] for k in k_values},
        "mrr": [],
        "ndcg": {k: [] for k in k_values},
        "hit_rate": {k: [] for k in k_values},
        "latency_ms": []
    }

    for item in test_data:
        query = item["query"]

        # Retrieve
        start = time.time()
        query_emb = embedder.encode_query(query)
        retrieved = vector_db.search(query_emb, top_k=max(k_values))
        latency = (time.time() - start) * 1000

        retrieved_ids = [str(r.get("metadata", {}).get("chunk_id", r.get("id", ""))) for r in retrieved]

        # For this test, relevant is the chunk with matching context
        relevant_ids = [f"chunk_{SAMPLE_EVAL_DATA.index(item)}"]

        # Metrics
        for k in k_values:
            results["precision"][k].append(evaluator.precision_at_k(retrieved_ids, relevant_ids, k))
            results["recall"][k].append(evaluator.recall_at_k(retrieved_ids, relevant_ids, k))
            results["ndcg"][k].append(evaluator.ndcg_at_k(retrieved_ids, relevant_ids, k))
            results["hit_rate"][k].append(evaluator.hit_rate_at_k(retrieved_ids, relevant_ids, k))

        results["mrr"].append(evaluator.mean_reciprocal_rank(retrieved_ids, relevant_ids))
        results["latency_ms"].append(latency)

    # Print results
    print(f"\nNumber of queries: {len(test_data)}")
    print("\n--- Metrics ---")

    for k in k_values:
        print(f"\n@{k}:")
        print(f"  Precision: {sum(results['precision'][k])/len(results['precision'][k]):.4f}")
        print(f"  Recall: {sum(results['recall'][k])/len(results['recall'][k]):.4f}")
        print(f"  NDCG: {sum(results['ndcg'][k])/len(results['ndcg'][k]):.4f}")
        print(f"  Hit Rate: {sum(results['hit_rate'][k])/len(results['hit_rate'][k]):.4f}")

    print(f"\nMRR: {sum(results['mrr'])/len(results['mrr']):.4f}")
    print(f"Avg Latency: {sum(results['latency_ms'])/len(results['latency_ms']):.2f}ms")

    return results


def evaluate_generation_metrics(embedder, test_data):
    """Danh gia generation metrics (F1, BLEU, Semantic Similarity)"""
    from modules.evaluation_module import RAGEvaluator

    evaluator = RAGEvaluator(embedder=embedder)

    print("\n" + "=" * 60)
    print("GENERATION METRICS EVALUATION")
    print("=" * 60)

    results = {
        "f1_score": [],
        "bleu_score": [],
        "semantic_similarity": [],
        "exact_match": []
    }

    for item in test_data:
        # Simulate prediction as context (in real scenario, this would be LLM output)
        prediction = item["context"]
        ground_truth = item["ground_truth_answer"]

        results["f1_score"].append(evaluator.f1_score(prediction, ground_truth))
        results["bleu_score"].append(evaluator.bleu_score(prediction, ground_truth))
        results["exact_match"].append(evaluator.exact_match(prediction, ground_truth))

        if embedder:
            results["semantic_similarity"].append(
                evaluator.semantic_similarity(prediction, ground_truth)
            )

    # Print results
    print(f"\nNumber of samples: {len(test_data)}")
    print("\n--- Metrics ---")
    print(f"  F1 Score: {sum(results['f1_score'])/len(results['f1_score']):.4f}")
    print(f"  BLEU Score: {sum(results['bleu_score'])/len(results['bleu_score']):.4f}")
    print(f"  Exact Match: {sum(results['exact_match'])/len(results['exact_match']):.4f}")

    if results["semantic_similarity"]:
        print(f"  Semantic Similarity: {sum(results['semantic_similarity'])/len(results['semantic_similarity']):.4f}")

    return results


def run_full_evaluation():
    """Chay danh gia day du"""
    from modules.embedding_module import TextEmbedding
    from modules.vector_db_module import VectorDatabase

    print("=" * 60)
    print("AUDIO IR SYSTEM - FULL EVALUATION")
    print("=" * 60)

    # 1. Initialize components
    print("\n[1/4] Khoi tao Embedding model...")
    embedder = TextEmbedding(provider="local", model_name="sbert")

    print("\n[2/4] Khoi tao Vector Database...")
    vector_db = VectorDatabase(
        collection_name="eval_collection",
        embedding_dimension=embedder.embedding_dim
    )

    # 2. Create and index test chunks
    print("\n[3/4] Tao va index test data...")
    chunks = create_test_chunks()

    # Encode chunks
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_text(texts, show_progress=False)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    # Add to vector db
    vector_db.add_documents(chunks)
    print(f"  Da index {len(chunks)} chunks")

    # 3. Run evaluations
    print("\n[4/4] Chay danh gia...")

    # Retrieval evaluation
    retrieval_results = evaluate_retrieval_only(
        embedder, vector_db, SAMPLE_EVAL_DATA, k_values=[1, 3, 5]
    )

    # Generation metrics evaluation
    generation_results = evaluate_generation_metrics(embedder, SAMPLE_EVAL_DATA)

    # 4. Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "embedding_model": embedder.model_name,
            "embedding_provider": embedder.provider,
            "num_test_samples": len(SAMPLE_EVAL_DATA)
        },
        "retrieval": {
            "mrr": sum(retrieval_results['mrr']) / len(retrieval_results['mrr']),
            "avg_latency_ms": sum(retrieval_results['latency_ms']) / len(retrieval_results['latency_ms']),
        },
        "generation": {
            "f1_score": sum(generation_results['f1_score']) / len(generation_results['f1_score']),
            "bleu_score": sum(generation_results['bleu_score']) / len(generation_results['bleu_score']),
        }
    }

    # Add k-specific metrics
    for k in [1, 3, 5]:
        results["retrieval"][f"precision@{k}"] = sum(retrieval_results['precision'][k]) / len(retrieval_results['precision'][k])
        results["retrieval"][f"recall@{k}"] = sum(retrieval_results['recall'][k]) / len(retrieval_results['recall'][k])
        results["retrieval"][f"ndcg@{k}"] = sum(retrieval_results['ndcg'][k]) / len(retrieval_results['ndcg'][k])
        results["retrieval"][f"hit_rate@{k}"] = sum(retrieval_results['hit_rate'][k]) / len(retrieval_results['hit_rate'][k])

    if generation_results["semantic_similarity"]:
        results["generation"]["semantic_similarity"] = sum(generation_results['semantic_similarity']) / len(generation_results['semantic_similarity'])

    output_file = output_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nKet qua da luu tai: {output_file}")

    # Cleanup
    vector_db.delete_collection()

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  MRR: {results['retrieval']['mrr']:.4f}")
    print(f"  Precision@5: {results['retrieval']['precision@5']:.4f}")
    print(f"  NDCG@5: {results['retrieval']['ndcg@5']:.4f}")
    print(f"  F1 Score: {results['generation']['f1_score']:.4f}")
    if 'semantic_similarity' in results['generation']:
        print(f"  Semantic Sim: {results['generation']['semantic_similarity']:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_full_evaluation()
