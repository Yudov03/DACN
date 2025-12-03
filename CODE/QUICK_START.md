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

Mở `.env` và thêm API key:

```env
# Google (miễn phí)
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

## 4. Test nhanh

```bash
python tests/test_new_modules.py
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

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `GOOGLE_API_KEY chua duoc cau hinh` | Thêm key vào `.env` |
| `CUDA out of memory` | Đổi `WHISPER_MODEL=tiny` |
| `UnicodeEncodeError` | Chạy `chcp 65001` |

---

Xem chi tiết tại [README.md](README.md)
