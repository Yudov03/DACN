"""
Demo script for Anti-Hallucination Modules
===========================================

Demonstrates:
- Answer Verification (grounding check)
- Abstention Checker (know when to say "I don't know")
- Conflict Detection (handle conflicting information)

Usage:
    python scripts/demo_anti_hallucination.py
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


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demo_answer_verification():
    """Demo Answer Verification module."""
    print_section("1. ANSWER VERIFICATION")

    from modules import AnswerVerifier

    verifier = AnswerVerifier()

    test_cases = [
        {
            "context": "Học phí năm 2024 là 15 triệu đồng mỗi kỳ. Sinh viên có thể đóng làm 2 đợt.",
            "question": "Học phí một kỳ là bao nhiêu?",
            "answer": "Học phí là 15 triệu đồng mỗi kỳ.",
            "expected": "FULLY_GROUNDED"
        },
        {
            "context": "Trường được thành lập năm 1956. Có 5 khoa chính.",
            "question": "Học phí là bao nhiêu?",
            "answer": "Học phí là 20 triệu đồng.",
            "expected": "LIKELY_HALLUCINATED"
        },
        {
            "context": "Machine learning cho phép máy tính học từ dữ liệu. Deep learning là một dạng ML.",
            "question": "ML là gì?",
            "answer": "Machine learning cho phép máy học từ dữ liệu và được ứng dụng trong nhiều lĩnh vực như y tế.",
            "expected": "PARTIALLY_GROUNDED"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Context: {case['context'][:80]}...")
        print(f"Question: {case['question']}")
        print(f"Answer: {case['answer']}")
        print(f"Expected: {case['expected']}")

        result = verifier.verify(
            answer=case['answer'],
            context=case['context'],
            question=case['question']
        )

        print(f"\nResult:")
        print(f"  Grounding: {result.grounding_level.value}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Should abstain: {result.should_abstain}")
        if result.suggested_action:
            print(f"  Suggested action: {result.suggested_action[:100]}...")

        status = "PASS" if result.grounding_level.value == case['expected'] else "CHECK"
        print(f"  Status: {status}")


def demo_abstention_checker():
    """Demo Abstention Checker module."""
    print_section("2. ABSTENTION CHECKER")

    from modules import AbstentionChecker

    checker = AbstentionChecker(min_retrieval_score=0.5)

    test_cases = [
        {
            "question": "Học phí là bao nhiêu?",
            "contexts": [
                {"similarity": 0.85, "text": "Học phí năm 2024 là 15 triệu đồng."},
                {"similarity": 0.72, "text": "Sinh viên đóng học phí theo kỳ."}
            ],
            "expected_abstain": False
        },
        {
            "question": "Điểm IELTS tối thiểu là bao nhiêu?",
            "contexts": [
                {"similarity": 0.25, "text": "Trường có nhiều chương trình đào tạo."},
                {"similarity": 0.18, "text": "Sinh viên cần đăng ký môn học."}
            ],
            "expected_abstain": True
        },
        {
            "question": "Thời gian nghỉ hè?",
            "contexts": [],
            "expected_abstain": True
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Question: {case['question']}")
        print(f"Contexts: {len(case['contexts'])} items")
        if case['contexts']:
            avg_sim = sum(c['similarity'] for c in case['contexts']) / len(case['contexts'])
            print(f"Avg similarity: {avg_sim:.2f}")
        print(f"Expected abstain: {case['expected_abstain']}")

        should_abstain, reason = checker.should_abstain(
            question=case['question'],
            retrieved_chunks=case['contexts']
        )

        print(f"\nResult:")
        print(f"  Should abstain: {should_abstain}")
        print(f"  Reason: {reason}")

        status = "PASS" if should_abstain == case['expected_abstain'] else "CHECK"
        print(f"  Status: {status}")


def demo_conflict_detection():
    """Demo Conflict Detection module."""
    print_section("3. CONFLICT DETECTION")

    from modules import ConflictDetector

    detector = ConflictDetector()

    test_cases = [
        {
            "name": "Date Conflict",
            "chunks": [
                {
                    "text": "Học phí năm 2023 là 12 triệu đồng.",
                    "metadata": {"date": "2023-01-01", "source": "old_doc.pdf"},
                    "similarity": 0.85
                },
                {
                    "text": "Học phí năm 2024 là 15 triệu đồng (quy định mới).",
                    "metadata": {"date": "2024-01-01", "source": "new_doc.pdf"},
                    "similarity": 0.82
                }
            ],
            "query": "học phí"
        },
        {
            "name": "No Conflict",
            "chunks": [
                {
                    "text": "Machine learning là một nhánh của AI.",
                    "metadata": {"source": "ml_intro.pdf"},
                    "similarity": 0.90
                },
                {
                    "text": "Deep learning là một dạng của machine learning.",
                    "metadata": {"source": "dl_intro.pdf"},
                    "similarity": 0.88
                }
            ],
            "query": "machine learning"
        },
        {
            "name": "Numeric Conflict",
            "chunks": [
                {
                    "text": "Điểm đạt tối thiểu là 5.0",
                    "metadata": {"date": "2022-06-01"},
                    "similarity": 0.80
                },
                {
                    "text": "Điểm đạt tối thiểu là 4.0 (áp dụng từ 2024)",
                    "metadata": {"date": "2024-01-01"},
                    "similarity": 0.78
                }
            ],
            "query": "điểm đạt"
        }
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Query: {case['query']}")
        print(f"Chunks: {len(case['chunks'])}")
        for chunk in case['chunks']:
            date = chunk['metadata'].get('date', 'N/A')
            print(f"  - [{date}] {chunk['text'][:50]}...")

        result = detector.detect_and_resolve(case['chunks'], case['query'])

        print(f"\nResult:")
        print(f"  Has conflicts: {result.has_conflicts}")
        if result.has_conflicts:
            print(f"  Conflicts found: {len(result.conflicts)}")
            if result.resolution_notes:
                print(f"  Resolution: {result.resolution_notes[0]}")
        if result.resolved_context:
            print(f"  Resolved context: {result.resolved_context[:100]}...")


def demo_integrated():
    """Demo integrated anti-hallucination flow."""
    print_section("4. INTEGRATED FLOW")

    from modules import AnswerVerifier, AbstentionChecker, ConflictDetector

    print("\nSimulating RAG with anti-hallucination...")

    # Initialize
    verifier = AnswerVerifier()
    abstention = AbstentionChecker(min_retrieval_score=0.5)
    conflict = ConflictDetector()

    # Scenario: Query with conflicting sources
    question = "Học phí một kỳ là bao nhiêu?"
    retrieved_chunks = [
        {
            "text": "Học phí 2023: 12 triệu đồng/kỳ",
            "metadata": {"date": "2023-01-01"},
            "similarity": 0.85
        },
        {
            "text": "Học phí 2024: 15 triệu đồng/kỳ (cập nhật)",
            "metadata": {"date": "2024-01-01"},
            "similarity": 0.82
        }
    ]
    generated_answer = "Học phí là 15 triệu đồng mỗi kỳ."

    print(f"\nQuestion: {question}")
    print(f"Retrieved: {len(retrieved_chunks)} chunks")
    print(f"Generated answer: {generated_answer}")

    # Step 1: Check abstention
    print("\n[Step 1] Abstention Check...")
    contexts = [{"similarity": c["similarity"], "text": c["text"]} for c in retrieved_chunks]
    should_abstain, reason = abstention.should_abstain(question, contexts)
    print(f"  Should abstain: {should_abstain}")
    if should_abstain:
        print(f"  STOP: {reason}")
        return

    # Step 2: Detect conflicts
    print("\n[Step 2] Conflict Detection...")
    conflict_result = conflict.detect_and_resolve(retrieved_chunks, question)
    print(f"  Has conflicts: {conflict_result.has_conflicts}")
    if conflict_result.has_conflicts and conflict_result.resolution_notes:
        print(f"  Resolution: {conflict_result.resolution_notes[0]}")

    # Step 3: Verify answer
    print("\n[Step 3] Answer Verification...")
    context = conflict_result.resolved_context or " ".join([c["text"] for c in retrieved_chunks])
    verify_result = verifier.verify(generated_answer, context, question)
    print(f"  Grounding: {verify_result.grounding_level.value}")
    print(f"  Confidence: {verify_result.confidence_score:.2f}")

    # Final decision
    print("\n[Final Decision]")
    if verify_result.confidence_score >= 0.7:
        print(f"  ACCEPT answer: {generated_answer}")
        if conflict_result.has_conflicts and conflict_result.resolution_notes:
            print(f"  Note: {conflict_result.resolution_notes[0]}")
    else:
        print(f"  REJECT answer (low confidence)")
        print(f"  Suggest: Ask for clarification or abstain")


def main():
    print("=" * 60)
    print("   ANTI-HALLUCINATION MODULES DEMO")
    print("=" * 60)

    demo_answer_verification()
    demo_abstention_checker()
    demo_conflict_detection()
    demo_integrated()

    print("\n" + "=" * 60)
    print("   DEMO COMPLETED!")
    print("=" * 60)
    print("\nModules demonstrated:")
    print("  - AnswerVerifier: Check if answer is grounded in context")
    print("  - AbstentionChecker: Know when to say 'I don't know'")
    print("  - ConflictDetector: Handle conflicting information")


if __name__ == "__main__":
    main()
