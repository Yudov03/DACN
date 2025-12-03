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
# Clone và tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình

```bash
# Copy config template
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
# Chọn 1 trong 2 provider:

# Option A: Google (recommended - free tier)
GOOGLE_API_KEY=your_google_api_key
LLM_PROVIDER=google
EMBEDDING_PROVIDER=google

# Option B: OpenAI
OPENAI_API_KEY=your_openai_api_key
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
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
│       └── rag_module.py         # RAG + LLM
│
├── tests/
│   └── test_new_modules.py       # Test suite
│
└── data/
    ├── audio/             # Audio input
    ├── transcripts/       # ASR output
    └── outputs/           # Results
```

## Modules

### 1. ASR Module - Whisper
Chuyển audio thành text với timestamps.

```python
from src.modules import WhisperASR

asr = WhisperASR(model_name="base")  # tiny, base, small, medium, large
transcript = asr.transcribe_audio("audio.mp3")
# Output: {"full_text": "...", "segments": [{"text": "...", "start": 0.0, "end": 5.0}]}
```

### 2. Chunking Module - LangChain
Chia văn bản thành chunks với timestamp preservation.

```python
from src.modules import TextChunker

chunker = TextChunker(
    chunk_size=500,
    chunk_overlap=50,
    method="recursive",  # fixed, sentence, recursive, semantic
    embedding_provider="google"  # cho semantic chunking
)
chunks = chunker.chunk_transcript(transcript)
```

### 3. Embedding Module - OpenAI/Google
Tạo vector embeddings.

```python
from src.modules import TextEmbedding

# Google (768 dimensions)
embedder = TextEmbedding(provider="google")

# OpenAI (1536 dimensions)
embedder = TextEmbedding(provider="openai")

embeddings = embedder.encode_chunks(chunks)
```

### 4. Vector Database - Qdrant
Lưu trữ và tìm kiếm vectors.

```python
from src.modules import VectorDatabase

# In-memory (development)
vector_db = VectorDatabase(
    collection_name="audio_transcripts",
    embedding_dimension=768  # 768 for Google, 1536 for OpenAI
)

# Docker (production)
vector_db = VectorDatabase(
    host="localhost",
    port=6333,
    collection_name="audio_transcripts"
)

vector_db.add_documents(chunks_with_embeddings)
results = vector_db.search(query_embedding, top_k=5)
```

### 5. RAG Module - LLM
Retrieval-Augmented Generation.

```python
from src.modules import RAGSystem

rag = RAGSystem(
    vector_db=vector_db,
    embedder=embedder,
    provider="google",  # hoặc "openai"
)

response = rag.query("Nội dung chính là gì?")
print(response["answer"])
# Kèm timestamps: [audio.mp3:00:01:30-00:02:15]
```

## Sử dụng

### Command Line

```bash
# Xử lý một file audio
python main.py --mode process --audio data/audio/lecture.mp3

# Xử lý cả thư mục
python main.py --mode process --audio data/audio/

# Query một lần
python main.py --mode query --question "AI là gì?"

# Interactive mode
python main.py --mode interactive
```

### Python API

```python
from main import AudioIRPipeline

# Initialize với Google
pipeline = AudioIRPipeline(
    llm_provider="google",
    embedding_provider="google"
)

# Process audio
pipeline.process_audio("podcast.mp3")

# Query
response = pipeline.query("Chủ đề chính là gì?")
print(response["answer"])

# Xem sources với timestamps
for src in response["sources"]:
    print(f"[{src['start_time_formatted']}] {src['text'][:100]}...")
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
LLM_PROVIDER=google          # google hoặc openai
EMBEDDING_PROVIDER=google    # google hoặc openai

# API Keys
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=audio_transcripts

# Whisper
WHISPER_MODEL=base           # tiny, base, small, medium, large
WHISPER_DEVICE=cuda          # cuda hoặc cpu

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNKING_METHOD=recursive    # fixed, sentence, recursive, semantic

# RAG
TOP_K=5
LLM_TEMPERATURE=0.7
```

## Testing

```bash
# Chạy test suite
python tests/test_new_modules.py

# Test output mong đợi:
# config: PASS
# chunking_basic: PASS
# qdrant_inmemory: PASS
# embedding: PASS
# pipeline_mock: PASS
```

## Qdrant Setup (Optional)

Mặc định hệ thống sử dụng **Qdrant in-memory** mode. Để persistent storage:

```bash
# Docker
docker run -p 6333:6333 qdrant/qdrant

# Hoặc Qdrant Cloud: https://cloud.qdrant.io
```

## Troubleshooting

### Lỗi API Key

```
ValueError: GOOGLE_API_KEY chua duoc cau hinh!
```
→ Thêm API key vào file `.env`

### Lỗi Out of Memory (Whisper)

```
RuntimeError: CUDA out of memory
```
→ Sử dụng model nhỏ hơn: `WHISPER_MODEL=tiny` hoặc `WHISPER_DEVICE=cpu`

### Lỗi Unicode (Windows)

```
UnicodeEncodeError: 'charmap' codec can't encode
```
→ Đã được fix trong code. Nếu vẫn lỗi, chạy: `chcp 65001`

## Tech Stack

- **ASR**: OpenAI Whisper
- **Chunking**: LangChain Text Splitters
- **Embedding**: OpenAI/Google via LangChain
- **Vector DB**: Qdrant
- **LLM**: OpenAI GPT / Google Gemini
- **Framework**: LangChain

## License

MIT License

---

**Đồ án chuyên nghành - 2025**
