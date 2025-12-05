"""
Run Evaluation Script - Chay danh gia he thong RAG
"""

import sys
import os
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(override=True)


def run_quick_evaluation():
    """Chay danh gia nhanh voi sample data"""

    print("=" * 60)
    print("RAG SYSTEM EVALUATION")
    print("=" * 60)

    from modules.embedding_module import TextEmbedding
    from modules.vector_db_module import VectorDatabase
    from modules.rag_module import RAGSystem
    from modules.evaluation_module import RAGEvaluator

    # Get provider
    provider = os.getenv("EMBEDDING_PROVIDER", "google")
    print(f"\nProvider: {provider.upper()}")

    # 1. Setup components
    print("\n[1/4] Khoi tao components...")
    embedder = TextEmbedding(provider=provider)

    dim = embedder.embedding_dim
    vector_db = VectorDatabase(
        collection_name="eval_test",
        embedding_dimension=dim
    )

    rag = RAGSystem(
        vector_db=vector_db,
        embedder=embedder,
        provider=os.getenv("LLM_PROVIDER", "google")
    )

    # 2. Add sample documents
    print("\n[2/4] Them sample documents...")
    sample_docs = [
        {
            "text": "Tri tue nhan tao (AI) la linh vuc khoa hoc may tinh tap trung vao viec tao ra cac he thong thong minh. AI co the hoc, suy luan va tu cai thien.",
            "chunk_id": 0,
            "doc_id": "doc_0"
        },
        {
            "text": "Machine Learning la mot nhanh quan trong cua AI. No cho phep may tinh hoc tu du lieu ma khong can duoc lap trinh cu the. Cac thuat toan ML bao gom supervised, unsupervised va reinforcement learning.",
            "chunk_id": 1,
            "doc_id": "doc_1"
        },
        {
            "text": "Deep Learning su dung cac mang neural nhieu tang de xu ly du lieu. No rat hieu qua trong nhan dien hinh anh, xu ly ngon ngu tu nhien va nhieu ung dung khac.",
            "chunk_id": 2,
            "doc_id": "doc_2"
        }
    ]

    chunks_with_emb = embedder.encode_chunks(sample_docs)
    vector_db.add_documents(chunks_with_emb)
    print(f"  Da them {len(sample_docs)} documents")

    # 3. Prepare test data
    print("\n[3/4] Chuan bi test data...")
    test_data = [
        {
            "query": "AI la gi?",
            "relevant_doc_ids": ["0"],  # chunk_id as string
            "ground_truth_answer": "AI la linh vuc khoa hoc may tinh tao ra cac he thong thong minh"
        },
        {
            "query": "Machine Learning hoat dong nhu the nao?",
            "relevant_doc_ids": ["1"],
            "ground_truth_answer": "Machine Learning cho phep may tinh hoc tu du lieu ma khong can lap trinh cu the"
        },
        {
            "query": "Deep Learning dung de lam gi?",
            "relevant_doc_ids": ["2"],
            "ground_truth_answer": "Deep Learning dung de nhan dien hinh anh va xu ly ngon ngu tu nhien"
        }
    ]
    print(f"  {len(test_data)} test cases")

    # 4. Run evaluation
    print("\n[4/4] Chay evaluation...")
    evaluator = RAGEvaluator(
        rag_system=rag,
        embedder=embedder,
        vector_db=vector_db
    )

    results = evaluator.evaluate_end_to_end(
        test_data,
        k_values=[1, 3, 5]
    )

    # Print results
    evaluator.print_results(results)

    # Save results
    output_path = Path("data/evaluation/results")
    output_path.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, output_path / "evaluation_results.json")

    # Cleanup
    vector_db.delete_collection()
    print("\nDa xoa test collection")

    return results


def run_full_evaluation(test_data_path: str):
    """Chay danh gia day du voi test dataset"""

    import json

    print("=" * 60)
    print("FULL RAG EVALUATION")
    print("=" * 60)

    # Load test data
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"\nLoaded {len(test_data)} test cases from {test_data_path}")

    # Similar to quick evaluation but with real data
    # ... implementation same as above but with loaded test_data

    print("\nTo run full evaluation, first process your audio files:")
    print("  python main.py --mode process --audio data/audio/")
    print("\nThen run this script with your test dataset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG Evaluation")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Evaluation mode"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/evaluation/test_dataset.json",
        help="Path to test dataset (for full mode)"
    )

    args = parser.parse_args()

    if args.mode == "quick":
        run_quick_evaluation()
    else:
        run_full_evaluation(args.test_data)
