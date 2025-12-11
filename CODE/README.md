# Audio Information Retrieval System

Hệ thống Truy xuất Thông tin từ Âm thanh sử dụng ASR (Whisper), Vector Database (Qdrant), và LLM (Ollama/OpenAI/Google Gemini).

## Kiến trúc

```
Audio → ASR (Whisper) → Chunking → Embedding → Qdrant → RAG + LLM → Answer
                                      ↓
                              [Optimizations]
                    Query Expansion | Context Compression
                    Caching | Better Prompts | Reranking
```

**Hỗ trợ nhiều providers:**
- **Local**: SBERT/E5 (Embedding) + Ollama (LLM) - Miễn phí, offline
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
# Option 1: Local (miễn phí, offline)
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=e5
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5

# Option 2: Google Cloud
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

# Web UI
streamlit run app.py
```

## Cấu trúc thư mục

```
CODE/
├── main.py                 # Entry point
├── app.py                  # Streamlit Web UI
├── requirements.txt        # Dependencies
├── .env.example           # Config template
│
├── src/
│   ├── config.py          # System config
│   └── modules/
│       ├── asr_module.py              # Whisper ASR
│       ├── chunking_module.py         # Text Splitter
│       ├── embedding_module.py        # SBERT/E5/OpenAI/Google
│       ├── vector_db_module.py        # Qdrant + BM25 Hybrid
│       ├── rag_module.py              # Ollama/GPT/Gemini
│       ├── reranker_module.py         # Cross-Encoder Reranking
│       ├── evaluation_module.py       # Metrics
│       │
│       │  # Optimization Modules
│       ├── query_expansion_module.py      # Query Expansion
│       ├── context_compression_module.py  # Context Compression
│       ├── caching_module.py              # Embedding/Response Cache
│       └── prompt_templates.py            # RAG Prompts
│
├── scripts/
│   ├── demo_optimizations.py     # Demo optimization modules
│   ├── evaluate_real_datasets.py # Evaluation with datasets
│   ├── download_dataset.py       # Download test datasets
│   └── run_benchmark.py          # Full benchmark
│
├── tests/
│   └── test_new_modules.py    # Test suite
│
└── data/
    ├── audio/                 # Audio input
    ├── transcripts/           # ASR output
    └── evaluation/
        ├── datasets/          # Test datasets
        └── results/           # Evaluation results
```

## Modules

### Core Modules

#### 1. ASR Module - Whisper
```python
from src.modules import WhisperASR

asr = WhisperASR(model_name="base")  # tiny, base, small, medium, large
transcript = asr.transcribe_audio("audio.mp3")
```

#### 2. Embedding Module - Local/Cloud
```python
from src.modules import TextEmbedding

# Local (recommended)
embedder = TextEmbedding(provider="local", model_name="e5")

# Cloud
embedder = TextEmbedding(provider="google")
embeddings = embedder.encode_chunks(chunks)
```

#### 3. Vector Database - Qdrant + Hybrid Search
```python
from src.modules import VectorDatabase

vector_db = VectorDatabase(collection_name="transcripts", embedding_dimension=768)

# Hybrid search (Vector + BM25)
results = vector_db.hybrid_search(
    query="machine learning",
    query_embedding=emb,
    alpha=0.7,  # 0.7 vector + 0.3 BM25
    top_k=5
)
```

#### 4. RAG Module - Ollama/GPT/Gemini
```python
from src.modules import RAGSystem

rag = RAGSystem(
    vector_db=vector_db,
    embedder=embedder,
    provider="ollama",  # or google, openai
)
response = rag.query("Nội dung chính là gì?")
```

#### 5. Reranker Module
```python
from src.modules import CrossEncoderReranker

reranker = CrossEncoderReranker()
results = vector_db.search_with_rerank(query, emb, reranker, top_k=5)
```

### Optimization Modules

#### 6. Query Expansion
```python
from src.modules import QueryExpander, MultiQueryRetriever

# Expand query với synonyms
expander = QueryExpander(method="synonym")
queries = expander.expand("AI là gì?")
# ['AI là gì?', 'trí tuệ nhân tạo là gì?', ...]

# Multi-query retrieval với RRF fusion
retriever = MultiQueryRetriever(vector_db, embedder, expander)
results = retriever.retrieve(query, top_k=5, fusion_method="rrf")
```

#### 7. Context Compression
```python
from src.modules import ContextCompressor

# Nén context giảm 60-75% tokens
compressor = ContextCompressor(method="extractive", max_tokens=500)
compressed, chunks = compressor.compress(query, contexts)
```

#### 8. Caching
```python
from src.modules import CacheManager

cache = CacheManager(cache_dir="./cache")

# Cache embeddings (~0.01ms per hit)
cache.set_embedding("text", "model", embedding)
cached = cache.get_embedding("text", "model")

# Cache LLM responses
cache.set_response(prompt, model, response)
```

#### 9. Prompt Templates
```python
from src.modules import PromptTemplateManager

manager = PromptTemplateManager(language="vi")
# Templates: basic_qa, audio_qa, factual_qa, cot_qa

sys_prompt, user_prompt = manager.format_prompt(
    "audio_qa",
    context=context,
    question=question
)
```

## Evaluation

### Run Evaluation

```bash
# Evaluate với datasets thực tế
python scripts/evaluate_real_datasets.py --dataset all --embedding e5

# Demo optimization modules
python scripts/demo_optimizations.py
```

### Results

#### Embedding Model Comparison
| Model | MRR | NDCG@5 | Latency |
|-------|-----|--------|---------|
| SBERT | 0.72 | 0.68 | 45ms |
| E5 | 0.89 | 0.85 | 52ms |
| E5-large | 0.91 | 0.87 | 78ms |

#### Search Method Comparison
| Method | MRR | Notes |
|--------|-----|-------|
| Vector only | 0.85 | Good for semantic |
| BM25 only | 0.78 | Good for keywords |
| Hybrid (0.7) | 0.89 | Best overall |
| + Reranking | 0.92 | Best quality |

## Cấu hình

### Models

| Type | Provider | Model | Dimensions |
|------|----------|-------|------------|
| Embedding | Local | SBERT | 768 |
| Embedding | Local | E5 | 768 |
| Embedding | Google | text-embedding-004 | 768 |
| Embedding | OpenAI | text-embedding-3-small | 1536 |
| LLM | Local | Ollama (qwen2.5) | - |
| LLM | Google | gemini-2.0-flash | - |
| LLM | OpenAI | gpt-4o-mini | - |

### Environment Variables

```env
# Provider Selection
EMBEDDING_PROVIDER=local    # local, google, openai
LLM_PROVIDER=ollama         # ollama, google, openai

# Local Models
LOCAL_EMBEDDING_MODEL=e5    # sbert, e5, e5-large
OLLAMA_MODEL=qwen2.5

# API Keys (nếu dùng cloud)
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
CHUNKING_METHOD=semantic

# RAG
TOP_K=5
LLM_TEMPERATURE=0.7
```

## Testing

```bash
# Chạy test suite
python tests/test_new_modules.py

# Test optimization modules
python scripts/demo_optimizations.py
```

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `API_KEY chưa được cấu hình` | Thêm key vào `.env` hoặc dùng local models |
| `CUDA out of memory` | Đổi `WHISPER_MODEL=tiny` |
| `UnicodeEncodeError` | Chạy `chcp 65001` |
| `Ollama connection refused` | Chạy `ollama serve` trước |
| `429 Rate limit exceeded` | Đợi 1 phút hoặc dùng local models |

## Tech Stack

- **ASR**: OpenAI Whisper
- **Embedding**: Sentence-BERT, E5, OpenAI, Google
- **Vector DB**: Qdrant + BM25 Hybrid
- **LLM**: Ollama, OpenAI GPT, Google Gemini
- **Reranking**: Cross-Encoder (sentence-transformers)
- **Optimization**: Query Expansion, Context Compression, Caching
- **Evaluation**: MRR, NDCG, Precision, Recall, F1, BLEU
- **Web UI**: Streamlit

## License

MIT License

---

**Đồ án chuyên ngành - 2025**
