"""
Demo script for Knowledge Base Module
======================================

Demonstrates:
- Adding documents to KB
- Searching documents
- Semantic search
- Export/Import
- Statistics

Usage:
    python scripts/demo_knowledge_base.py
    python scripts/demo_knowledge_base.py --folder data/documents/
"""

import sys
import io
import tempfile
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demo_basic_operations():
    """Demo basic KB operations."""
    print_section("1. BASIC OPERATIONS")

    from modules import KnowledgeBase

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create KB
        print("\n--- Create Knowledge Base ---")
        kb = KnowledgeBase(base_dir=tmpdir)
        print(f"KB created at: {tmpdir}")

        # Create sample documents
        doc1_path = Path(tmpdir) / "ai_intro.txt"
        doc1_path.write_text("""
        Artificial Intelligence (AI) Introduction

        AI is the simulation of human intelligence by machines.
        Machine learning is a subset of AI.
        Deep learning uses neural networks with many layers.
        """, encoding="utf-8")

        doc2_path = Path(tmpdir) / "ml_guide.txt"
        doc2_path.write_text("""
        Machine Learning Guide

        There are three types of machine learning:
        1. Supervised Learning - learns from labeled data
        2. Unsupervised Learning - finds patterns in unlabeled data
        3. Reinforcement Learning - learns through rewards
        """, encoding="utf-8")

        # Add documents
        print("\n--- Add Documents ---")
        doc_id1 = kb.add_document(str(doc1_path), tags=["ai", "introduction"])
        print(f"Added: {doc1_path.name} -> {doc_id1}")

        doc_id2 = kb.add_document(str(doc2_path), tags=["ml", "guide"])
        print(f"Added: {doc2_path.name} -> {doc_id2}")

        # Get stats
        print("\n--- Statistics ---")
        stats = kb.get_stats()
        print(f"Total documents: {stats.total_documents}")
        print(f"Total chunks: {stats.total_chunks}")
        print(f"Total size: {stats.total_size_bytes:,} bytes")

        # List documents
        print("\n--- List Documents ---")
        docs = kb.list_documents()
        for doc in docs:
            print(f"  - {doc.get('filename', 'N/A')} [{doc.get('id', 'N/A')[:8]}...]")


def demo_search():
    """Demo KB search functionality."""
    print_section("2. SEARCH")

    from modules import KnowledgeBase

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(base_dir=tmpdir)

        # Add sample documents
        docs = [
            ("regulations.txt", "Quy định học phí năm 2024 là 15 triệu đồng mỗi kỳ.", ["hocphi", "2024"]),
            ("attendance.txt", "Sinh viên cần đạt 80% điểm danh để được thi.", ["diemdanh", "thi"]),
            ("grading.txt", "Điểm thi cuối kỳ chiếm 60%, giữa kỳ chiếm 40%.", ["diem", "thi"]),
            ("schedule.txt", "Thời gian đăng ký môn học là 2 tuần đầu học kỳ.", ["dangky", "monhoc"]),
        ]

        for filename, content, tags in docs:
            doc_path = Path(tmpdir) / filename
            doc_path.write_text(content, encoding="utf-8")
            kb.add_document(str(doc_path), tags=tags)

        print(f"Added {len(docs)} documents")

        # Search by filename
        print("\n--- Search by Filename ---")
        query = "regulations"
        results = kb.search_documents(query)
        print(f"Query: '{query}'")
        print(f"Results: {len(results)}")
        for r in results:
            print(f"  - {r.get('filename', 'N/A')}")

        # Search by tag
        print("\n--- Search by Tag ---")
        tag = "thi"
        results = kb.search_documents(f"tag:{tag}")
        print(f"Tag: '{tag}'")
        print(f"Results: {len(results)}")
        for r in results:
            print(f"  - {r.get('filename', 'N/A')}: {r.get('tags', [])}")

        # Semantic search
        print("\n--- Semantic Search ---")
        query = "học phí bao nhiêu tiền"
        results = kb.semantic_search(query, top_k=3)
        print(f"Query: '{query}'")
        print(f"Results: {len(results)}")
        for r in results:
            sim = r.get('similarity', 0)
            text = r.get('text', '')[:50]
            print(f"  - [{sim:.3f}] {text}...")


def demo_export_import():
    """Demo export/import functionality."""
    print_section("3. EXPORT / IMPORT")

    from modules import KnowledgeBase

    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir1 = Path(tmpdir) / "kb1"
        kb_dir1.mkdir()
        kb_dir2 = Path(tmpdir) / "kb2"
        kb_dir2.mkdir()
        export_path = Path(tmpdir) / "backup.zip"

        # Create first KB with documents
        print("\n--- Create KB with Documents ---")
        kb1 = KnowledgeBase(base_dir=str(kb_dir1))

        doc_path = Path(tmpdir) / "test.txt"
        doc_path.write_text("This is a test document for export.", encoding="utf-8")
        kb1.add_document(str(doc_path), tags=["test"])

        stats1 = kb1.get_stats()
        print(f"KB1 documents: {stats1.total_documents}")

        # Export
        print("\n--- Export ---")
        kb1.export_kb(str(export_path))
        print(f"Exported to: {export_path}")
        print(f"File size: {export_path.stat().st_size:,} bytes")

        # Import to new KB
        print("\n--- Import to New KB ---")
        kb2 = KnowledgeBase(base_dir=str(kb_dir2))
        kb2.import_kb(str(export_path))

        stats2 = kb2.get_stats()
        print(f"KB2 documents: {stats2.total_documents}")
        print(f"Import successful: {stats1.total_documents == stats2.total_documents}")


def demo_with_real_documents(folder_path: str):
    """Demo with real documents from a folder."""
    print_section("4. REAL DOCUMENTS")

    from modules import KnowledgeBase, UnifiedProcessor

    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return

    processor = UnifiedProcessor()
    supported = processor.supported_extensions()

    # Find supported files
    files = []
    for ext in supported:
        files.extend(folder.glob(f"*{ext}"))

    if not files:
        print(f"No supported files in: {folder_path}")
        return

    print(f"\nFound {len(files)} files")

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(base_dir=tmpdir)

        # Add files
        print("\n--- Adding Documents ---")
        for i, file_path in enumerate(files[:5], 1):  # Limit to 5 for demo
            print(f"  [{i}] {file_path.name}...", end=" ")
            try:
                doc_id = kb.add_document(str(file_path))
                print(f"OK ({doc_id[:8]}...)")
            except Exception as e:
                print(f"ERROR: {e}")

        # Stats
        print("\n--- Statistics ---")
        stats = kb.get_stats()
        print(f"Documents: {stats.total_documents}")
        print(f"Chunks: {stats.total_chunks}")

        # Search
        print("\n--- Semantic Search ---")
        queries = ["học phí", "machine learning", "quy định"]
        for query in queries:
            results = kb.semantic_search(query, top_k=2)
            print(f"\nQuery: '{query}'")
            for r in results[:2]:
                sim = r.get('similarity', 0)
                text = r.get('text', '')[:60]
                print(f"  [{sim:.3f}] {text}...")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo Knowledge Base Module")
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder with real documents"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("   KNOWLEDGE BASE MODULE DEMO")
    print("=" * 60)

    demo_basic_operations()
    demo_search()
    demo_export_import()

    if args.folder:
        demo_with_real_documents(args.folder)

    print("\n" + "=" * 60)
    print("   DEMO COMPLETED!")
    print("=" * 60)
    print("\nKnowledge Base features:")
    print("  - Add/remove documents (34 formats)")
    print("  - Search by filename, tags")
    print("  - Semantic search")
    print("  - Export/Import backup")
    print("  - Statistics")


if __name__ == "__main__":
    main()
