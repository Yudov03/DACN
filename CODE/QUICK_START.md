# Quick Start Guide

## Cài đặt nhanh (5 phút)

### 1. Clone/Download code

```bash
cd CODE
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình

Tạo file `.env`:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Mở file `.env` và thêm OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

## Sử dụng nhanh

### Bước 1: Thêm audio files

Đặt file audio (mp3, wav, m4a) vào thư mục `data/audio/`

### Bước 2: Xử lý audio

```bash
python main.py --mode process --audio data/audio/your-file.mp3
```

Hoặc xử lý tất cả files trong thư mục:

```bash
python main.py --mode process --audio data/audio/
```

### Bước 3: Truy vấn

**Chế độ interactive (recommended):**

```bash
python main.py --mode interactive
```

Sau đó nhập câu hỏi:

```
💬 Câu hỏi của bạn: Nội dung chính của audio là gì?
```

**Hoặc query trực tiếp:**

```bash
python main.py --mode query --question "Nội dung chính là gì?"
```

## Ví dụ đầy đủ

```bash
# 1. Xử lý audio
python main.py --mode process --audio data/audio/podcast.mp3

# Output:
# [1/4] Transcribing audio...
# [2/4] Chunking transcript...
# [3/4] Creating embeddings...
# [4/4] Storing in vector database...
# ✓ Hoàn thành! Đã xử lý và lưu 25 chunks

# 2. Query
python main.py --mode interactive

# 💬 Câu hỏi của bạn: Chủ đề chính là gì?
#
# ANSWER:
# Chủ đề chính của audio là về trí tuệ nhân tạo và ứng dụng của nó...
#
# SOURCES (5 chunks):
# [Source 1] Similarity: 0.8234
# Audio: podcast.mp3
# Time: 00:02:15.00 - 00:03:45.00
# Text: Trí tuệ nhân tạo đang thay đổi nhiều lĩnh vực...
```

## Commands cheat sheet

```bash
# Xử lý 1 file audio
python main.py --mode process --audio data/audio/file.mp3

# Xử lý nhiều files
python main.py --mode process --audio data/audio/

# Query một lần
python main.py --mode query --question "Câu hỏi?"

# Query với nhiều kết quả hơn
python main.py --mode query --question "Câu hỏi?" --top-k 10

# Interactive mode
python main.py --mode interactive

# Trong interactive mode:
# - Gõ câu hỏi để query
# - Gõ "stats" để xem thống kê
# - Gõ "exit" để thoát
```

## Troubleshooting nhanh

**Q: "OPENAI_API_KEY chưa được cấu hình"**
→ Thêm API key vào file `.env`

**Q: Out of memory**
→ Sửa trong `.env`: `WHISPER_MODEL=tiny` hoặc `base`

**Q: Không tìm thấy kết quả**
→ Kiểm tra đã xử lý audio chưa: `python main.py --mode process --audio <file>`

**Q: Kết quả không chính xác**
→ Thử tăng TOP_K: `--top-k 10`

## Next steps

1. Đọc [README.md](README.md) để hiểu chi tiết hơn
2. Xem [examples/example_usage.py](examples/example_usage.py) để biết cách dùng nâng cao
3. Tùy chỉnh các tham số trong `.env` hoặc `src/config.py`

## Support

Nếu gặp lỗi, kiểm tra:
1. Python version >= 3.8
2. Đã cài đặt đầy đủ dependencies
3. OpenAI API key hợp lệ
4. Đủ dung lượng disk cho models và data

---

**Chúc bạn sử dụng thành công!** 🚀
