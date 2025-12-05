# Audio Information Retrieval System

Hệ thống Truy xuất Thông tin từ Âm thanh sử dụng ASR (Whisper), Vector Database (Qdrant), và LLM (OpenAI/Google Gemini).

## Kiến trúc

```
Audio → ASR (Whisper) → Chunking (LangChain) → Embedding → Qdrant → RAG + LLM → Answer
```

**Hỗ trợ 2 providers:**
- **Google**: Gemini 2.0 Flash + Text Embedding 004
- **OpenAI**: GPT-4o-mini + Text Embedding 3

## Quick Start

### 1. Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
# Google (recommended - free tier)
GOOGLE_API_KEY=your_google_api_key
LLM_PROVIDER=google
EMBEDDING_PROVIDER=google
```

### 3. Chạy

```bash
# Xử lý audio
python main.py --mode process --audio data/audio/sample.mp3

# Query
python main.py --mode interactive
```

## Cấu trúc thư mục

```
CODE/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env.example           # Config template
│
├── src/
│   ├── config.py          # System config
│   └── modules/
│       ├── asr_module.py         # Whisper ASR
│       ├── chunking_module.py    # LangChain Text Splitter
│       ├── embedding_module.py   # OpenAI/Google Embeddings
│       ├── vector_db_module.py   # Qdrant Vector DB
│       ├── rag_module.py         # RAG + LLM
│       └── evaluation_module.py  # Benchmark & Metrics
│
├── scripts/
│   ├── download_dataset.py    # Download SQuAD, Vietnamese QA
│   ├── run_benchmark.py       # Full benchmark
│   ├── run_evaluation.py      # Quick evaluation
│   └── tune_parameters.py     # Parameter tuning
│
├── tests/
│   └── test_new_modules.py    # Test suite
│
└── data/
    ├── audio/                 # Audio input
    ├── transcripts/           # ASR output
    ├── evaluation/
    │   ├── datasets/          # Test datasets
    │   └── benchmark_results/ # Results
    └── tuning_results/        # Tuning results
```

## Modules

### 1. ASR Module - Whisper
```python
from src.modules import WhisperASR

asr = WhisperASR(model_name="base")  # tiny, base, small, medium, large
transcript = asr.transcribe_audio("audio.mp3")
```

### 2. Chunking Module - LangChain
```python
from src.modules import TextChunker

chunker = TextChunker(
    chunk_size=500,
    chunk_overlap=50,
    method="recursive",  # fixed, sentence, recursive, semantic
)
chunks = chunker.chunk_transcript(transcript)
```

### 3. Embedding Module - OpenAI/Google
```python
from src.modules import TextEmbedding

embedder = TextEmbedding(provider="google")  # hoặc "openai"
embeddings = embedder.encode_chunks(chunks)
```

### 4. Vector Database - Qdrant
```python
from src.modules import VectorDatabase

vector_db = VectorDatabase(
    collection_name="audio_transcripts",
    embedding_dimension=768  # 768 for Google, 1536 for OpenAI
)
vector_db.add_documents(chunks_with_embeddings)
results = vector_db.search(query_embedding, top_k=5)
```

### 5. RAG Module - LLM
```python
from src.modules import RAGSystem

rag = RAGSystem(
    vector_db=vector_db,
    embedder=embedder,
    provider="google",
)
response = rag.query("Nội dung chính là gì?")
print(response["answer"])
```

### 6. Evaluation Module - Metrics
```python
from src.modules import RAGEvaluator

evaluator = RAGEvaluator(
    rag_system=rag,
    embedder=embedder,
    vector_db=vector_db
)

# Retrieval metrics: Precision@K, Recall@K, MRR, NDCG
# Generation metrics: F1, BLEU, Semantic Similarity
results = evaluator.evaluate_end_to_end(test_data, k_values=[1, 3, 5, 10])
```

## Benchmark & Fine-tuning

### Download Datasets

```bash
# Vietnamese QA dataset
python scripts/download_dataset.py --dataset vietnamese

# SQuAD 2.0 (English)
python scripts/download_dataset.py --dataset squad --samples 50
```

### Run Benchmark

```bash
# Benchmark với Vietnamese dataset
python scripts/run_benchmark.py --dataset vietnamese

# Benchmark với SQuAD
python scripts/run_benchmark.py --dataset squad
```

Output:
```
RETRIEVAL METRICS:
  MRR, Precision@K, Recall@K, NDCG@K, Hit Rate@K

GENERATION METRICS:
  F1 Score, BLEU Score, Semantic Similarity, Latency
```

### Parameter Tuning

```bash
# Random search (nhanh)
python scripts/tune_parameters.py --method random --iterations 10

# Grid search (kỹ hơn)
python scripts/tune_parameters.py --method grid
```

Tham số có thể tune:
- `chunk_size`: 300, 500, 800
- `chunk_overlap`: 30, 50, 100
- `chunking_method`: fixed, sentence, recursive
- `top_k`: 3, 5, 10
- `llm_temperature`: 0.3, 0.7

### Best Config (từ tuning):

```python
{
    "chunk_size": 500,
    "chunk_overlap": 50,
    "chunking_method": "sentence",
    "top_k": 5,
    "llm_temperature": 0.3
}
```

## Testing

```bash
# Chạy test suite
python tests/test_new_modules.py

# Expected output:
# config: PASS
# chunking_basic: PASS
# qdrant_inmemory: PASS
# embedding: PASS
# pipeline_mock: PASS
# Total: 5 passed, 0 failed
```

## Cấu hình

### Models

| Provider | Embedding Model | Dimensions | LLM Model |
|----------|-----------------|------------|-----------|
| Google | text-embedding-004 | 768 | gemini-2.0-flash |
| OpenAI | text-embedding-3-small | 1536 | gpt-4o-mini |

### Environment Variables

```env
# Provider Selection
LLM_PROVIDER=google
EMBEDDING_PROVIDER=google

# API Keys
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Whisper
WHISPER_MODEL=base
WHISPER_DEVICE=cuda

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNKING_METHOD=recursive

# RAG
TOP_K=5
LLM_TEMPERATURE=0.7
```

## Qdrant Setup (Optional)

Mặc định sử dụng **Qdrant in-memory**. Để persistent storage:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `GOOGLE_API_KEY chua duoc cau hinh` | Thêm key vào `.env` |
| `CUDA out of memory` | Đổi `WHISPER_MODEL=tiny` |
| `UnicodeEncodeError` | Chạy `chcp 65001` |
| `429 Rate limit exceeded` | Đợi 1 phút hoặc dùng paid tier |

## Tech Stack

- **ASR**: OpenAI Whisper
- **Chunking**: LangChain Text Splitters
- **Embedding**: OpenAI/Google via LangChain
- **Vector DB**: Qdrant
- **LLM**: OpenAI GPT / Google Gemini
- **Evaluation**: Custom metrics (MRR, NDCG, F1, BLEU)

## License

MIT License

---

**Đồ án chuyên nghành - 2025**
