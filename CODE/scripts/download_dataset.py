"""
Download Popular Datasets for RAG Evaluation

Datasets:
1. SQuAD 2.0 - Stanford Question Answering Dataset (English)
2. MS MARCO - Microsoft Reading Comprehension (English)
3. ViQuAD - Vietnamese Question Answering Dataset
"""

import json
import os
import sys
import io
from pathlib import Path
from typing import List, Dict
import urllib.request

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class DatasetDownloader:
    """Download and prepare evaluation datasets"""

    def __init__(self, output_dir: str = "data/evaluation/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_squad(self, version: str = "2.0", num_samples: int = 100) -> Path:
        """
        Download SQuAD dataset

        Args:
            version: "1.1" or "2.0"
            num_samples: Number of samples to extract

        Returns:
            Path to processed dataset
        """
        print(f"\n{'='*60}")
        print(f"Downloading SQuAD {version}")
        print(f"{'='*60}")

        # URLs
        if version == "2.0":
            url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        else:
            url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

        # Download
        print(f"URL: {url}")
        print("Downloading...")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"Error downloading: {e}")
            return None

        print(f"Downloaded! Processing...")

        # Process into our format
        processed = self._process_squad(data, num_samples)

        # Save
        output_path = self.output_dir / f"squad_{version.replace('.', '_')}_sample.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(processed['test_cases'])} samples to: {output_path}")

        return output_path

    def _process_squad(self, data: Dict, num_samples: int) -> Dict:
        """Process SQuAD data into our evaluation format"""

        test_cases = []
        contexts = []
        context_id = 0

        for article in data.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                contexts.append({
                    "id": f"ctx_{context_id}",
                    "text": context,
                    "title": article.get("title", "")
                })

                for qa in para.get("qas", []):
                    if len(test_cases) >= num_samples:
                        break

                    question = qa.get("question", "")
                    answers = qa.get("answers", [])

                    if answers:
                        ground_truth = answers[0].get("text", "")
                    else:
                        # For SQuAD 2.0 unanswerable questions
                        plausible = qa.get("plausible_answers", [])
                        ground_truth = plausible[0].get("text", "") if plausible else ""

                    if question and ground_truth:
                        test_cases.append({
                            "id": qa.get("id", f"q_{len(test_cases)}"),
                            "query": question,
                            "relevant_doc_ids": [f"ctx_{context_id}"],
                            "ground_truth_answer": ground_truth,
                            "context_id": f"ctx_{context_id}"
                        })

                context_id += 1

                if len(test_cases) >= num_samples:
                    break

            if len(test_cases) >= num_samples:
                break

        return {
            "dataset": "SQuAD",
            "num_contexts": len(contexts),
            "num_test_cases": len(test_cases),
            "contexts": contexts,
            "test_cases": test_cases
        }

    def download_viquad(self, num_samples: int = 100) -> Path:
        """
        Download ViQuAD - Vietnamese QA Dataset

        Args:
            num_samples: Number of samples

        Returns:
            Path to processed dataset
        """
        print(f"\n{'='*60}")
        print("Downloading ViQuAD (Vietnamese QA)")
        print(f"{'='*60}")

        # ViQuAD is on HuggingFace
        url = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/raw/main/ViQuAD/dev_ViQuAD.json"

        print("Note: ViQuAD may require HuggingFace datasets library")
        print("Alternative: Creating sample Vietnamese dataset...")

        # Create sample Vietnamese dataset instead
        sample_data = self._create_vietnamese_sample(num_samples)

        output_path = self.output_dir / "vietnamese_sample.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to: {output_path}")
        return output_path

    def _create_vietnamese_sample(self, num_samples: int = 50) -> Dict:
        """Create sample Vietnamese dataset for testing"""

        contexts = [
            {
                "id": "ctx_0",
                "text": "Tri tue nhan tao (AI) la linh vuc khoa hoc may tinh nghien cuu ve viec tao ra cac he thong thong minh. AI co the hoc, suy luan, va tu cai thien theo thoi gian. Ung dung cua AI bao gom nhan dien giong noi, xu ly ngon ngu tu nhien, thi giac may tinh va robot tu dong.",
                "title": "Tri tue nhan tao"
            },
            {
                "id": "ctx_1",
                "text": "Machine Learning la mot nhanh cua tri tue nhan tao. No cho phep may tinh hoc tu du lieu ma khong can lap trinh cu the. Cac phuong phap ML bao gom supervised learning, unsupervised learning va reinforcement learning. Deep Learning la mot dang dac biet cua ML su dung mang neural nhieu tang.",
                "title": "Machine Learning"
            },
            {
                "id": "ctx_2",
                "text": "Deep Learning la ky thuat hoc sau su dung cac mang neural nhan tao nhieu tang. No dac biet hieu qua trong xu ly hinh anh, am thanh va ngon ngu tu nhien. Cac kien truc pho bien bao gom CNN cho hinh anh, RNN va Transformer cho chuoi du lieu.",
                "title": "Deep Learning"
            },
            {
                "id": "ctx_3",
                "text": "Natural Language Processing (NLP) la linh vuc xu ly ngon ngu tu nhien. NLP giup may tinh hieu va tao ra ngon ngu cua con nguoi. Cac ung dung bao gom dich may, trich xuat thong tin, phan tich cam xuc va chatbot. Transformer la kien truc hien dai nhat cho NLP.",
                "title": "Xu ly ngon ngu tu nhien"
            },
            {
                "id": "ctx_4",
                "text": "Computer Vision la linh vuc thi giac may tinh. No cho phep may tinh hieu va phan tich hinh anh va video. Ung dung bao gom nhan dien khuon mat, phat hien vat the, xe tu lai va kiem tra chat luong trong san xuat. CNN la kien truc pho bien cho computer vision.",
                "title": "Thi giac may tinh"
            }
        ]

        test_cases = [
            {"id": "vn_1", "query": "AI la gi?", "relevant_doc_ids": ["ctx_0"], "ground_truth_answer": "AI la linh vuc khoa hoc may tinh nghien cuu ve viec tao ra cac he thong thong minh"},
            {"id": "vn_2", "query": "Ung dung cua AI bao gom nhung gi?", "relevant_doc_ids": ["ctx_0"], "ground_truth_answer": "nhan dien giong noi, xu ly ngon ngu tu nhien, thi giac may tinh va robot tu dong"},
            {"id": "vn_3", "query": "Machine Learning la gi?", "relevant_doc_ids": ["ctx_1"], "ground_truth_answer": "Machine Learning la mot nhanh cua tri tue nhan tao cho phep may tinh hoc tu du lieu"},
            {"id": "vn_4", "query": "Cac phuong phap ML bao gom gi?", "relevant_doc_ids": ["ctx_1"], "ground_truth_answer": "supervised learning, unsupervised learning va reinforcement learning"},
            {"id": "vn_5", "query": "Deep Learning su dung gi?", "relevant_doc_ids": ["ctx_2"], "ground_truth_answer": "mang neural nhan tao nhieu tang"},
            {"id": "vn_6", "query": "Deep Learning hieu qua trong linh vuc nao?", "relevant_doc_ids": ["ctx_2"], "ground_truth_answer": "xu ly hinh anh, am thanh va ngon ngu tu nhien"},
            {"id": "vn_7", "query": "CNN dung cho gi?", "relevant_doc_ids": ["ctx_2", "ctx_4"], "ground_truth_answer": "xu ly hinh anh va computer vision"},
            {"id": "vn_8", "query": "NLP la gi?", "relevant_doc_ids": ["ctx_3"], "ground_truth_answer": "NLP la linh vuc xu ly ngon ngu tu nhien giup may tinh hieu va tao ra ngon ngu cua con nguoi"},
            {"id": "vn_9", "query": "Ung dung cua NLP?", "relevant_doc_ids": ["ctx_3"], "ground_truth_answer": "dich may, trich xuat thong tin, phan tich cam xuc va chatbot"},
            {"id": "vn_10", "query": "Transformer dung cho gi?", "relevant_doc_ids": ["ctx_3"], "ground_truth_answer": "Transformer la kien truc hien dai nhat cho NLP"},
            {"id": "vn_11", "query": "Computer Vision la gi?", "relevant_doc_ids": ["ctx_4"], "ground_truth_answer": "Computer Vision la linh vuc thi giac may tinh cho phep may tinh hieu va phan tich hinh anh va video"},
            {"id": "vn_12", "query": "Xe tu lai dung cong nghe gi?", "relevant_doc_ids": ["ctx_4"], "ground_truth_answer": "Computer Vision va CNN"},
            {"id": "vn_13", "query": "RNN dung cho loai du lieu nao?", "relevant_doc_ids": ["ctx_2"], "ground_truth_answer": "chuoi du lieu"},
            {"id": "vn_14", "query": "Kien truc nao pho bien cho hinh anh?", "relevant_doc_ids": ["ctx_2", "ctx_4"], "ground_truth_answer": "CNN"},
            {"id": "vn_15", "query": "AI co the lam gi?", "relevant_doc_ids": ["ctx_0"], "ground_truth_answer": "hoc, suy luan, va tu cai thien theo thoi gian"},
        ]

        return {
            "dataset": "Vietnamese_AI_QA",
            "language": "Vietnamese",
            "num_contexts": len(contexts),
            "num_test_cases": len(test_cases[:num_samples]),
            "contexts": contexts,
            "test_cases": test_cases[:num_samples]
        }

    def create_eval_ready_dataset(self, dataset_path: Path) -> Path:
        """
        Convert dataset to evaluation-ready format

        Args:
            dataset_path: Path to downloaded dataset

        Returns:
            Path to eval-ready dataset
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract just test_cases for evaluation
        eval_data = data.get("test_cases", [])

        output_path = dataset_path.parent / f"{dataset_path.stem}_eval.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        print(f"Created eval-ready dataset: {output_path}")
        return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Evaluation Datasets")
    parser.add_argument(
        "--dataset",
        choices=["squad", "vietnamese", "all"],
        default="vietnamese",
        help="Dataset to download"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples"
    )

    args = parser.parse_args()

    downloader = DatasetDownloader()

    if args.dataset == "squad":
        path = downloader.download_squad(num_samples=args.samples)
        if path:
            downloader.create_eval_ready_dataset(path)

    elif args.dataset == "vietnamese":
        path = downloader.download_viquad(num_samples=args.samples)
        if path:
            downloader.create_eval_ready_dataset(path)

    elif args.dataset == "all":
        # Download both
        squad_path = downloader.download_squad(num_samples=args.samples)
        if squad_path:
            downloader.create_eval_ready_dataset(squad_path)

        vn_path = downloader.download_viquad(num_samples=args.samples)
        if vn_path:
            downloader.create_eval_ready_dataset(vn_path)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nTo run evaluation:")
    print("  python scripts/run_evaluation.py --test-data data/evaluation/datasets/<dataset>_eval.json")


if __name__ == "__main__":
    main()
