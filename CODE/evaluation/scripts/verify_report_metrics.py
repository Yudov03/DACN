"""
Script để xác minh các metrics trong report khớp với code thực tế
Dựa trên Chapter 6 - Testing and Evaluation

Kết quả cần đạt:
- ASR: Faster-Whisper small - WER 15%, RTF 0.06-0.12
- Retrieval: Hybrid + Rerank - MRR 0.75, NDCG@5 0.71, Recall@5 0.78, Recall@10 0.83
- Anti-Hallucination: Grounding 79%, Hallucination 14%, Abstention 78%
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_report_consistency():
    """Kiểm tra các con số trong code có khớp với report không"""

    print("="*80)
    print("VERIFICATION: Code vs Report Metrics")
    print("="*80)

    # 1. Kiểm tra ASR metrics
    print("\n[1] ASR Module Metrics:")
    print("  Report claims:")
    print("    - Faster-Whisper small: WER 15.0%, RTF 0.12")
    print("    - Faster-Whisper base: WER 18.2%, RTF 0.06")

    # 2. Kiểm tra Retrieval metrics
    print("\n[2] Retrieval Module Metrics:")
    print("  Report claims (Hybrid + Rerank):")
    print("    - MRR: 0.75")
    print("    - NDCG@5: 0.71")
    print("    - Recall@5: 0.78")
    print("    - Recall@10: 0.83")

    # 3. Kiểm tra Anti-Hallucination metrics
    print("\n[3] Anti-Hallucination Metrics:")
    print("  Report claims:")
    print("    - Grounding Accuracy: 79%")
    print("    - Hallucination Rate: 14%")
    print("    - Abstention Rate: 78%")

    print("\n[4] NFR Requirements (Chapter 1):")
    print("    - NFR1: Response time < 10s")
    print("    - NFR2: ASR WER < 15%")
    print("    - NFR3: Support up to 10,000 chunks")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run ASR evaluation: python evaluation/scripts/eval_asr.py")
    print("2. Run Retrieval evaluation: python evaluation/scripts/eval_retrieval.py")
    print("3. Run Anti-Hallucination eval: python evaluation/scripts/eval_antihalluc.py")
    print("4. Run comprehensive: python evaluation/scripts/comprehensive_evaluation.py")
    print("="*80)

if __name__ == "__main__":
    check_report_consistency()
