# Quick Start Guide

## 1. Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt
```

## 2. Cấu hình

```bash
cp .env.example .env
```

Mở `.env` và chọn một trong hai options:

### Option 1: Local (Miễn phí, Offline) - Recommended
```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

Cần cài Ollama: https://ollama.com/download
```bash
ollama pull qwen2.5:7b
```

### Option 2: Google Cloud
```env
GOOGLE_API_KEY=your_google_api_key
LLM_PROVIDER=google
EMBEDDING_PROVIDER=google
```

Lấy Google API key tại: https://aistudio.google.com/apikey

## 3. Import tài liệu

### Option A: All-in-One (Recommended)
```bash
# Đặt tài liệu vào data/resource/
# Import tất cả
python scripts/import_resources.py

# Xem trước files sẽ import
python scripts/import_resources.py --dry-run

# Reset và import lại từ đầu
python scripts/import_resources.py --clear
```

### Option B: Two-Step (Advanced)
```bash
# Step 1: Process OCR/ASR → RAW content
python scripts/process_resources.py

# Step 2: Post-process + Index → Qdrant
python scripts/reindex_documents.py

# Re-index một file cụ thể
python scripts/reindex_documents.py --file doc_id
```

**Khi nào dùng Two-Step?**
- Thay đổi embedding/chunking → chỉ cần `reindex_documents.py --reset`
- Debug OCR/ASR → kiểm tra `data/knowledge_base/processed/`
- Re-index một file → `reindex_documents.py --file doc_id`

## 4. Chạy ứng dụng

### Student Portal (Recommended):
```bash
streamlit run app.py
# http://localhost:8501
```
Sinh viên tra cứu thông tin, app tự động start Ollama.

### Admin Portal:
```bash
streamlit run app_admin.py --server.port 8502
# http://localhost:8502
```
Quản trị viên upload/quản lý tài liệu.

### CLI Mode:
```bash
python main.py --mode interactive
python main.py --mode query --question "Học phí bao nhiêu?"
python main.py --mode stats
```

## 5. Test nhanh

```bash
# Run all tests
python tests/run_tests.py

# Quick unit tests only
python tests/run_tests.py quick
```

## Ví dụ Python

```python
from src.modules import RAGSystem, TextEmbedding, VectorDatabase

# Khởi tạo components
embedder = TextEmbedding(provider="local")
vector_db = VectorDatabase(collection_name="knowledge_base")
rag = RAGSystem(vector_db=vector_db, embedder=embedder, provider="ollama")

# Hỏi đáp
response = rag.query("Nội dung chính là gì?")
print(response["answer"])
```

## Post-Processing với Cache

```python
from src.modules import PostProcessor

# Post-processing tự động cache kết quả
pp = PostProcessor(method="ollama")
result = pp.process(raw_text)  # Cache HIT ~0.0s, Cache MISS ~5-30s
```

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `API_KEY chưa được cấu hình` | Thêm key vào `.env` hoặc dùng local |
| `CUDA out of memory` | Đổi `WHISPER_MODEL=tiny` trong `.env` |
| `UnicodeEncodeError` | Chạy `chcp 65001` trước |
| `Ollama connection refused` | App tự động start Ollama, hoặc chạy `ollama serve` |
| `FFmpeg not found` | Cài FFmpeg: `winget install ffmpeg` |
| `PaddleOCR crash` | Đặt `OCR_MAX_IMAGE_SIZE=3500` trong `.env` |

---

Xem chi tiết tại [README.md](README.md)
