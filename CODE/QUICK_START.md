# Quick Start Guide

## 1. Cài đặt (2 phút)

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt
```

## 2. Cấu hình (1 phút)

```bash
cp .env.example .env
```

Mở `.env` và chọn một trong hai options:

### Option 1: Local (Miễn phí, Offline)
```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=e5
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5
```

Cần cài Ollama: https://ollama.ai/download
```bash
ollama pull qwen2.5
ollama serve
```

### Option 2: Google Cloud
```env
GOOGLE_API_KEY=your_google_api_key
LLM_PROVIDER=google
EMBEDDING_PROVIDER=google
```

Lấy Google API key tại: https://aistudio.google.com/apikey

## 3. Chạy

### Xử lý audio:
```bash
python main.py --mode process --audio data/audio/sample.mp3
```

### Query:
```bash
python main.py --mode interactive
```

### Web UI:
```bash
streamlit run app.py
```

## 4. Test nhanh

```bash
# Test modules
python tests/test_new_modules.py

# Test optimization modules
python scripts/demo_optimizations.py
```

Output mong đợi:
```
config: PASS
chunking_basic: PASS
qdrant_inmemory: PASS
embedding: PASS
pipeline_mock: PASS
```

## Ví dụ Python

```python
from main import AudioIRPipeline

# Khởi tạo
pipeline = AudioIRPipeline()

# Xử lý audio
pipeline.process_audio("audio.mp3")

# Hỏi đáp
response = pipeline.query("Nội dung chính là gì?")
print(response["answer"])
```

## Sử dụng Optimization Modules

```python
from src.modules import (
    QueryExpander,
    ContextCompressor,
    CacheManager,
    PromptTemplateManager
)

# Query Expansion
expander = QueryExpander(method="synonym")
queries = expander.expand("AI là gì?")

# Context Compression
compressor = ContextCompressor(method="extractive")
compressed, _ = compressor.compress(query, contexts)

# Caching
cache = CacheManager()
cache.set_embedding("text", "model", embedding)

# Prompt Templates
prompts = PromptTemplateManager(language="vi")
sys_p, user_p = prompts.format_prompt("audio_qa", context, question)
```

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `API_KEY chưa được cấu hình` | Thêm key vào `.env` hoặc dùng local |
| `CUDA out of memory` | Đổi `WHISPER_MODEL=tiny` |
| `UnicodeEncodeError` | Chạy `chcp 65001` |
| `Ollama connection refused` | Chạy `ollama serve` trước |

---

Xem chi tiết tại [README.md](README.md)
