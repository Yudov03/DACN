# Audio Information Retrieval System

Hệ thống Truy xuất Thông tin từ Âm thanh và Tài liệu sử dụng ASR (Whisper), Document Processing, Vector Database (Qdrant), LLM (Ollama/OpenAI/Google Gemini), và Text-to-Speech.

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                                    │
├────────────────────────┬────────────────────────────────────────────────┤
│   Audio/Video          │           Documents (68 formats)               │
│   (.mp3, .mp4, .wav)   │   (.pdf, .docx, .xlsx, .pptx, .html, etc.)    │
│         │              │               │                                 │
│         ▼              │               ▼                                 │
│   ASR (Whisper)        │     Document Processor                          │
│         │              │     (PDF/DOCX/Excel/OCR)                        │
│         └──────────────┴───────────────┬────────────────────────────────┤
│                                        ▼                                 │
│                              Text Chunking                               │
│                                    │                                     │
│                                    ▼                                     │
│                    Embedding (SBERT/E5/OpenAI/Google)                    │
│                                    │                                     │
│                                    ▼                                     │
│                     Vector Database (Qdrant + BM25)                      │
│                                    │                                     │
├────────────────────────────────────┼────────────────────────────────────┤
│                         OPTIMIZATIONS                                    │
│   Query Expansion | Context Compression | Caching | Reranking            │
├────────────────────────────────────┼────────────────────────────────────┤
│                                    ▼                                     │
│                    ENHANCED RAG + ANTI-HALLUCINATION                     │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │  Conflict Detection → Answer Verification → Safe Abstention │      │
│    │  (Date-aware)         (Grounding check)    (Low confidence) │      │
│    └─────────────────────────────────────────────────────────────┘      │
│                                    │                                     │
│                                    ▼                                     │
│                          RAG + LLM Generation                            │
│                      (Ollama/GPT/Gemini)                                 │
│                                    │                                     │
│                                    ▼                                     │
│                         Answer + TTS Output                              │
│                    (Text-to-Speech với giọng Việt)                       │
└─────────────────────────────────────────────────────────────────────────┘
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

### 2. Cài đặt Ollama (LLM local - miễn phí)

```bash
# Tải và cài đặt từ: https://ollama.com/download
# Sau khi cài xong, pull model:
ollama pull llama3.2

# Hoặc model tốt hơn cho tiếng Việt:
ollama pull qwen2.5
```

### 3. Cấu hình

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
# Option 1: Local (miễn phí, offline) - RECOMMENDED
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Option 2: Google Cloud
GOOGLE_API_KEY=your_google_api_key
LLM_PROVIDER=google
EMBEDDING_PROVIDER=google
```

### 4. Import tài liệu

```bash
# Đặt tài liệu vào data/resource/

# === Option A: All-in-One (Recommended) ===
python scripts/import_resources.py              # Import tất cả
python scripts/import_resources.py --clear      # Xóa và import lại
python scripts/import_resources.py --dry-run    # Xem trước

# === Option B: Two-Step (Advanced) ===
python scripts/process_resources.py             # Step 1: Process OCR/ASR
python scripts/reindex_documents.py             # Step 2: Index to Qdrant
python scripts/reindex_documents.py --file doc_id  # Re-index single file
```

**Two-Step workflow** hữu ích khi:
- Thay đổi embedding/chunking config → chỉ cần chạy `reindex_documents.py --reset`
- Debug OCR/ASR → kiểm tra `data/knowledge_base/processed/`
- Re-index một file → `reindex_documents.py --file doc_id`

### 5. Chạy ứng dụng

```bash
# Web UI cho sinh viên (tự động start Ollama)
streamlit run app.py

# CLI interactive mode
python main.py --mode interactive
```

**Features (app.py - Student Portal):**
- 💬 **Chat**: Hỏi đáp với Knowledge Base
- 🔍 **Search**: Tìm kiếm semantic
- 📚 **Sources**: Hiển thị nguồn tham khảo
- 🔊 **TTS**: Text-to-Speech (Vietnamese)
- ⚡ **Auto-start**: Tự động khởi động Ollama server

## Cấu trúc thư mục

```
CODE/
├── main.py                 # Entry point (CLI)
├── app.py                  # DocChat Platform (Web UI)
├── requirements.txt        # Dependencies
├── .env.example            # Config template
│
├── src/
│   ├── config.py           # System config
│   └── modules/
│       │  # Core Modules
│       ├── asr_module.py               # Whisper ASR
│       ├── chunking_module.py          # Text Splitter
│       ├── embedding_module.py         # SBERT/E5/OpenAI/Google
│       ├── vector_db_module.py         # Qdrant + BM25 Hybrid
│       ├── rag_module.py               # Enhanced RAG
│       ├── reranker_module.py          # Cross-Encoder Reranking
│       ├── evaluation_module.py        # Metrics
│       │
│       │  # Anti-Hallucination Modules
│       ├── answer_verification.py      # Grounding check + abstention
│       ├── conflict_detection.py       # Date-aware conflict resolution
│       │
│       │  # Document Processing (34 formats)
│       ├── document_processor/
│       │   ├── base.py                 # Base processor classes
│       │   ├── pdf_processor.py        # PDF extraction + OCR
│       │   ├── docx_processor.py       # Word document processor
│       │   ├── excel_processor.py      # Excel spreadsheets
│       │   ├── pptx_processor.py       # PowerPoint presentations
│       │   ├── text_processor.py       # Plain text processor
│       │   ├── audio_processor.py      # Audio files (Whisper)
│       │   ├── video_processor.py      # Video files (FFmpeg)
│       │   └── unified_processor.py    # Auto-detect processor
│       │
│       │  # Knowledge Base
│       ├── knowledge_base.py           # Document management
│       │
│       │  # Text-to-Speech
│       ├── tts_module.py               # TTS with edge-tts
│       │
│       │  # Optimization Modules
│       ├── query_expansion_module.py       # Query Expansion
│       ├── context_compression_module.py   # Context Compression
│       ├── caching_module.py               # Embedding/Response Cache
│       └── prompt_templates.py             # RAG Prompts (9 templates)
│
├── scripts/                # Admin + Demo scripts
│   ├── import_resources.py         # All-in-one import
│   ├── process_resources.py        # Step 1: Process OCR/ASR
│   ├── reindex_documents.py        # Step 2: Index to Qdrant
│   ├── demo_rag_pipeline.py        # RAG pipeline demo
│   ├── demo_document_processor.py  # Document processing demo
│   └── demo_anti_hallucination.py  # Anti-hallucination demo
│
├── evaluation/             # System evaluation
│   ├── datasets/           # Test datasets (Vietnamese, SQuAD)
│   ├── scripts/            # Evaluation scripts
│   │   ├── evaluate_system.py          # Basic evaluation
│   │   ├── evaluate_real_datasets.py   # Real datasets evaluation
│   │   ├── run_benchmark.py            # Full benchmark
│   │   ├── run_evaluation.py           # Quick/full evaluation
│   │   ├── tune_parameters.py          # Parameter tuning
│   │   └── download_dataset.py         # Dataset downloader
│   ├── results/            # Evaluation results
│   ├── benchmark_results/  # Benchmark results
│   └── tuning_results/     # Parameter tuning results
│
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── run_tests.py        # Test runner script
│   ├── test_unit.py        # Unit tests (43 tests)
│   ├── test_integration.py # Integration tests (12 tests)
│   ├── test_e2e.py         # E2E tests (9 tests)
│   └── test_data/          # Test data files
│
└── data/                   # Runtime data storage
    ├── resource/           # INPUT: Upload documents here
    │   ├── documents/      # PDF, DOCX, XLSX, Images, Code...
    │   └── audio/          # MP3, WAV, MP4, AVI...
    ├── knowledge_base/     # PROCESSED: System-managed
    │   ├── index.json      # Document registry
    │   ├── documents/      # Copied files
    │   ├── transcripts/    # ASR output
    │   └── processed/      # Processed JSON (RAW content)
    └── cache/              # CACHE: Post-processing cache
        └── post_processing/
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

### Anti-Hallucination Modules

#### 6. Answer Verification
```python
from src.modules import AnswerVerifier, AbstentionChecker

# Verify answer is grounded in context
verifier = AnswerVerifier()
result = verifier.verify(
    answer="Học phí là 15 triệu",
    context="Học phí năm 2024 là 15 triệu đồng/kỳ",
    question="Học phí bao nhiêu?"
)

print(result.grounding_level)    # FULLY_GROUNDED / PARTIALLY_GROUNDED / LIKELY_HALLUCINATED
print(result.confidence_score)   # 0.0 - 1.0
print(result.explanation)        # Chi tiết đánh giá

# Check if should abstain from answering
checker = AbstentionChecker(min_retrieval_score=0.5)
should_abstain, reason = checker.should_abstain(
    question="Điểm thi IELTS?",
    retrieved_contexts=[{"similarity": 0.3}]  # Low relevance
)
# should_abstain = True, reason = "No relevant context found"
```

**Grounding Levels:**
| Level | Description |
|-------|-------------|
| FULLY_GROUNDED | Tất cả claims có trong context |
| PARTIALLY_GROUNDED | Một số claims có trong context |
| LIKELY_HALLUCINATED | Claims không có trong context |

#### 7. Conflict Detection
```python
from src.modules import ConflictDetector

detector = ConflictDetector()

# Detect conflicts between chunks
chunks = [
    {"text": "Học phí 2023: 15 triệu", "metadata": {"date": "2023-01-01"}},
    {"text": "Học phí 2024: 18 triệu (mới)", "metadata": {"date": "2024-01-01"}},
]

result = detector.detect_and_resolve(chunks, "học phí")

print(result.has_conflicts)       # True
print(result.conflict_summary)    # "Found version conflicts"
print(result.recommended_chunks)  # Chunks mới nhất được ưu tiên
print(result.resolution_note)     # "Using latest information from 2024"
```

**Conflict Types:**
| Type | Detection |
|------|-----------|
| Date/Version | Ưu tiên thông tin mới nhất |
| Numeric | So sánh giá trị số |
| Semantic | Phát hiện mâu thuẫn ngữ nghĩa |

### Document Processing Modules

#### 8. Document Processor - 34 Formats
```python
from src.modules import UnifiedProcessor

# Auto-detect và xử lý document
processor = UnifiedProcessor()
doc = processor.process("document.pdf")

print(doc.content)       # Extracted text
print(doc.chunks)        # Text chunks với metadata
print(doc.tables)        # Extracted tables (PDF)
print(doc.metadata)      # Document metadata
```

**Supported formats (68 extensions):**

| Category | Formats |
|----------|---------|
| **Documents** | .pdf, .docx, .doc |
| **Presentations** | .pptx, .ppt |
| **Spreadsheets** | .xlsx, .xls |
| **Text/Data** | .txt, .md, .csv, .tsv, .json, .xml, .html, .log, .ini, .cfg, .rtf |
| **Code** | .py, .js, .ts, .jsx, .tsx, .java, .kt, .cpp, .c, .h, .hpp, .go, .rs, .rb, .php, .swift, .cs, .vb, .sql, .sh, .bash, .ps1, .yaml, .yml, .toml, .r, .R, .scala |
| **Audio** | .mp3, .wav, .m4a, .flac, .ogg, .wma, .aac |
| **Video** | .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v |
| **Images (OCR)** | .png, .jpg, .jpeg, .bmp, .tiff, .tif, .webp |

**Audio/Video Processing với timestamps:**
```python
from src.modules import UnifiedProcessor, format_transcript_with_timestamps

processor = UnifiedProcessor()

# Process video lecture
doc = processor.process("lecture.mp4")

# Get transcript with timestamps
for chunk in doc.chunks:
    start = chunk.metadata.get("start_time", 0)
    end = chunk.metadata.get("end_time", 0)
    print(f"[{start:.1f}s - {end:.1f}s] {chunk.text}")

# Or use helper function
print(format_transcript_with_timestamps(doc.chunks))

# Metadata includes duration, resolution, etc.
print(doc.metadata.extra)
# {'duration_seconds': 3600, 'resolution': '1920x1080', ...}
```

#### 9. Knowledge Base - Document Management
```python
from src.modules import KnowledgeBase

# Tạo Knowledge Base
kb = KnowledgeBase(base_dir="./kb_data")

# Thêm document
doc_id = kb.add_document("report.pdf", tags=["report", "2024"])

# Tìm kiếm (by filename/tags)
results = kb.search_documents("report")

# Semantic search (by content)
results = kb.semantic_search("machine learning applications", top_k=5)

# Export/Import
kb.export_kb("backup.zip")
kb.import_kb("backup.zip")

# Statistics
stats = kb.get_stats()
print(f"Documents: {stats.total_documents}")
print(f"Chunks: {stats.total_chunks}")
```

#### 10. Text-to-Speech (TTS) Module
```python
from src.modules import TextToSpeech, text_to_speech

# Simple function
audio_path = text_to_speech("Xin chào!", voice="vi-female")

# Full control với class
tts = TextToSpeech(voice="vi-female")
tts.set_rate("+10%")  # Faster
tts.set_volume("+20%")

# Synchronous synthesis
audio_bytes = tts.synthesize_sync("Nội dung cần đọc")

# Async synthesis (for streaming)
import asyncio
audio = asyncio.run(tts.synthesize("Async text"))

# Save to file
tts.save_to_file("output.mp3", "Text content")
```

**Available voices:**
| Voice | Language | Gender |
|-------|----------|--------|
| vi-female | Vietnamese | Female |
| vi-male | Vietnamese | Male |
| en-female | English | Female |
| en-male | English | Male |

### Optimization Modules

#### 11. Query Expansion
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

#### 12. Context Compression
```python
from src.modules import ContextCompressor

# Nén context giảm 60-75% tokens
compressor = ContextCompressor(method="extractive", max_tokens=500)
compressed, chunks = compressor.compress(query, contexts)
```

#### 13. Caching
```python
from src.modules import CacheManager

cache = CacheManager(cache_dir="./cache")

# Cache embeddings (~0.01ms per hit)
cache.set_embedding("text", "model", embedding)
cached = cache.get_embedding("text", "model")

# Cache LLM responses
cache.set_response(prompt, model, response)
```

#### 14. Prompt Templates (9 Templates)
```python
from src.modules import PromptTemplateManager

manager = PromptTemplateManager(language="vi")

# List available templates
templates = manager.list_templates()
# ['basic_qa', 'audio_qa', 'factual_qa', 'cot_qa',
#  'strict_qa', 'citation_required', 'conflict_aware',
#  'safe_abstention', 'summarize']

sys_prompt, user_prompt = manager.format_prompt(
    "strict_qa",  # Anti-hallucination template
    context=context,
    question=question
)
```

**Available Templates:**
| Template | Use Case |
|----------|----------|
| basic_qa | General Q&A |
| audio_qa | Audio transcripts with timestamps |
| factual_qa | Factual questions |
| cot_qa | Chain-of-thought reasoning |
| **strict_qa** | **Anti-hallucination (only answer from context)** |
| **citation_required** | **Must cite sources** |
| **conflict_aware** | **Handle conflicting information** |
| **safe_abstention** | **Say "I don't know" when uncertain** |
| summarize | Document summarization |

## Evaluation

### Run Evaluation

```bash
# Quick evaluation
python evaluation/scripts/run_evaluation.py --mode quick

# Evaluate với datasets thực tế
python evaluation/scripts/evaluate_real_datasets.py --dataset all --embedding e5 --save

# Full benchmark
python evaluation/scripts/run_benchmark.py --dataset vietnamese

# Parameter tuning
python evaluation/scripts/tune_parameters.py --method random --iterations 20

# Demo optimization modules
python scripts/demo_optimizations.py
```

See `evaluation/README.md` for more details.

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

Tất cả cấu hình được đọc từ file `.env`:

```env
# === Provider Selection ===
LLM_PROVIDER=ollama              # ollama, google, openai
EMBEDDING_PROVIDER=local         # local, google, openai

# === Ollama (Local LLM) ===
OLLAMA_MODEL=llama3.2            # llama3.2, qwen2.5, mistral, etc.
OLLAMA_BASE_URL=http://localhost:11434

# === Local Embedding ===
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LOCAL_EMBEDDING_DIMENSION=768
LOCAL_EMBEDDING_DEVICE=cuda      # cuda, cpu, or auto-detect

# === Cloud API Keys (optional) ===
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key

# === Qdrant Vector Database ===
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=knowledge_base

# === Whisper ASR ===
WHISPER_MODEL=base               # tiny, base, small, medium, large
WHISPER_DEVICE=cuda              # cuda or cpu

# === Chunking ===
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNKING_METHOD=semantic         # semantic, recursive, fixed, sentence
SEMANTIC_THRESHOLD=0.65
SEMANTIC_WINDOW_SIZE=5

# === RAG Parameters ===
TOP_K=5
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500
```

**Lưu ý:** App tự động đọc từ `.env`, không cần sửa code khi thay đổi cấu hình.

## Testing

```bash
# Chạy tất cả tests
python tests/run_tests.py              # All tests (unit + integration + e2e)
python tests/run_tests.py quick        # Quick unit tests only

# Hoặc dùng pytest
pytest tests/ -v                       # All tests
pytest tests/test_unit.py -v           # Unit tests (43 tests)
pytest tests/test_integration.py -v    # Integration tests (12 tests)
pytest tests/test_e2e.py -v            # E2E tests (9 tests)
```

### Test Structure

```
tests/
├── conftest.py          # Pytest fixtures
├── run_tests.py         # Test runner script
├── test_unit.py         # Unit tests (43 tests) - Individual modules
├── test_integration.py  # Integration tests (12 tests) - Pipelines
├── test_e2e.py          # E2E tests (9 tests) - Full workflows
└── test_data/           # Test data files
```

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_unit.py` | **43** | Unit tests cho từng module riêng lẻ |
| `test_integration.py` | **12** | Integration tests cho pipelines |
| `test_e2e.py` | **9** | End-to-end tests cho full workflows |

### Test Coverage (64 Tests)

**Unit Tests (43):**
- ✅ Chunking (fixed, sentence, recursive)
- ✅ Embedding (local SBERT/E5, similarity)
- ✅ VectorDB (init, add, search, stats)
- ✅ Document Processor (34 formats)
- ✅ Knowledge Base (init, add, stats)
- ✅ TTS (voices, synthesis, settings)
- ✅ Answer Verification (grounding, abstention)
- ✅ Conflict Detection (date extraction)
- ✅ Prompt Templates (9 templates)
- ✅ RAG (enhanced features)

**Integration Tests (12):**
- ✅ Chunking → Embedding pipeline
- ✅ Embedding → VectorDB pipeline
- ✅ Full retrieval pipeline
- ✅ Document → KB pipeline
- ✅ Anti-hallucination pipeline
- ✅ TTS integration
- ✅ Prompt integration

**E2E Tests (9):**
- ✅ Document to Answer flow
- ✅ Knowledge Base workflow
- ✅ Anti-hallucination workflow
- ✅ TTS output workflow
- ✅ Multi-format workflow
- ✅ System health check

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `Ollama connection refused` | App tự động start Ollama. Nếu không được: cài Ollama từ https://ollama.com/download rồi chạy `ollama pull llama3.2` |
| `API_KEY chưa được cấu hình` | Thêm key vào `.env` hoặc dùng local models (recommended) |
| `CUDA out of memory` | Đổi `WHISPER_MODEL=tiny` trong `.env` |
| `UnicodeEncodeError` | Chạy `chcp 65001` trước khi chạy script |
| `429 Rate limit exceeded` | Đợi 1 phút hoặc dùng local models |
| `FFmpeg not found` | Cài FFmpeg: `winget install ffmpeg` (Windows) hoặc `brew install ffmpeg` (Mac). Restart terminal. |
| `Video processing slow` | Dùng `WHISPER_MODEL=tiny` hoặc `base` |
| `I/O operation on closed file` | Streamlit bug - đã được fix trong app.py |
| `torch.classes warning` | Warning vô hại, đã được suppress |
| `OCR không chính xác` | Dùng PDF digital thay vì scan, hoặc ảnh chất lượng cao |

## Tech Stack

- **ASR**: OpenAI Whisper (Audio/Video transcription)
- **Document Processing**: PyMuPDF, python-docx, EasyOCR, pdfplumber, openpyxl, python-pptx
- **Video Processing**: FFmpeg (audio extraction), moviepy (fallback)
- **Embedding**: Sentence-BERT, E5, OpenAI, Google
- **Vector DB**: Qdrant + BM25 Hybrid
- **LLM**: Ollama, OpenAI GPT, Google Gemini
- **Reranking**: Cross-Encoder (sentence-transformers)
- **Anti-Hallucination**: Answer Verification, Conflict Detection, Safe Abstention
- **TTS**: edge-tts (Vietnamese + English voices)
- **Optimization**: Query Expansion, Context Compression, Caching, Prompt Templates
- **Evaluation**: MRR, NDCG, Precision, Recall, F1, BLEU
- **Web UI**: Streamlit
- **Testing**: pytest, comprehensive test suite (64 tests)

## License

MIT License

---

**Đồ án chuyên ngành - 2025**
