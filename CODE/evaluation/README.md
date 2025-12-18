# Evaluation Module

System evaluation for the Audio Information Retrieval system.

## Directory Structure

```
evaluation/
├── datasets/           # Test datasets
│   ├── vietnamese_sample.json       # Vietnamese QA dataset
│   ├── vietnamese_sample_eval.json  # Eval-ready format
│   ├── squad_2_0_sample.json        # SQuAD 2.0 dataset
│   ├── squad_2_0_sample_eval.json   # Eval-ready format
│   └── test_dataset.json            # Custom test dataset
├── scripts/            # Evaluation scripts
│   ├── evaluate_system.py           # Basic system evaluation
│   ├── evaluate_real_datasets.py    # Real datasets evaluation
│   ├── run_benchmark.py             # RAG benchmark
│   ├── run_evaluation.py            # Quick/full evaluation
│   ├── tune_parameters.py           # Parameter tuning
│   └── download_dataset.py          # Dataset downloader
├── results/            # Evaluation results (JSON)
├── benchmark_results/  # Benchmark results
└── tuning_results/     # Parameter tuning results
```

## Quick Start

### 1. Run Quick Evaluation

```bash
python evaluation/scripts/run_evaluation.py --mode quick
```

### 2. Evaluate with Real Datasets

```bash
# Vietnamese dataset
python evaluation/scripts/evaluate_real_datasets.py --dataset vietnamese --embedding e5 --save

# SQuAD dataset
python evaluation/scripts/evaluate_real_datasets.py --dataset squad --embedding sbert --save

# All datasets with reranker
python evaluation/scripts/evaluate_real_datasets.py --dataset all --reranker --save
```

### 3. Run Full Benchmark

```bash
python evaluation/scripts/run_benchmark.py --dataset vietnamese
```

### 4. Parameter Tuning

```bash
# Random search (faster)
python evaluation/scripts/tune_parameters.py --method random --iterations 20

# Grid search (thorough)
python evaluation/scripts/tune_parameters.py --method grid
```

### 5. Download More Datasets

```bash
python evaluation/scripts/download_dataset.py --dataset all --samples 100
```

## Metrics

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| **MRR** | Mean Reciprocal Rank - average of inverse rank of first correct result |
| **Precision@K** | Fraction of retrieved documents that are relevant |
| **Recall@K** | Fraction of relevant documents that are retrieved |
| **NDCG@K** | Normalized Discounted Cumulative Gain |
| **Hit Rate@K** | Fraction of queries with at least one relevant result |

### Generation Metrics
| Metric | Description |
|--------|-------------|
| **F1 Score** | Token-level F1 between prediction and ground truth |
| **BLEU Score** | N-gram overlap score |
| **Exact Match** | Exact string match after normalization |
| **Semantic Similarity** | Cosine similarity of embeddings |

## Dataset Format

Datasets should follow this JSON structure:

```json
{
    "dataset": "Dataset Name",
    "num_contexts": 10,
    "num_test_cases": 50,
    "contexts": [
        {
            "id": "ctx_0",
            "text": "Context text...",
            "title": "Optional title"
        }
    ],
    "test_cases": [
        {
            "id": "q_0",
            "query": "Question text?",
            "relevant_doc_ids": ["ctx_0"],
            "ground_truth_answer": "Expected answer"
        }
    ]
}
```

## Benchmark Results

Example results on Vietnamese dataset:

| Method | MRR | NDCG@5 | Latency |
|--------|-----|--------|---------|
| Vector Search | 0.72 | 0.68 | 45ms |
| BM25 | 0.58 | 0.52 | 12ms |
| Hybrid (alpha=0.7) | 0.89 | 0.85 | 52ms |
| With Reranker | 0.92 | 0.88 | 180ms |

## Adding Custom Datasets

1. Create a JSON file following the dataset format above
2. Place it in `evaluation/datasets/`
3. Run evaluation:

```bash
python evaluation/scripts/run_evaluation.py --mode full --test-data evaluation/datasets/your_dataset.json
```
