# Hệ thống Truy xuất Thông tin Đa phương thức từ Âm thanh

## Tổng quan

Hệ thống Information Retrieval kết hợp ASR (Automatic Speech Recognition), Text Embedding, Vector Database và Large Language Models (LLM) để truy xuất thông tin từ audio files.

**Giai đoạn 1: Text-Based IR từ Audio** (8 tuần đầu)

### Kiến trúc hệ thống

```
Audio Input
    ↓
ASR (Whisper) → Transcript + Timestamps
    ↓
Text Chunking (Semantic/Sentence-based)
    ↓
Text Embedding (Sentence-BERT/E5)
    ↓
Vector Database (ChromaDB/FAISS)
    ↓
Retrieval + RAG với LLM
    ↓
Answer + Timestamps
```

## Cài đặt

### 1. Yêu cầu hệ thống

- Python 3.8+
- CUDA (optional, cho GPU acceleration)
- 8GB+ RAM (16GB recommended cho Whisper models lớn hơn)

### 2. Cài đặt dependencies

```bash
# Tạo virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### 3. Cấu hình

Tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

Chỉnh sửa file `.env`:

```env
# OpenAI API Key (bắt buộc cho LLM)
OPENAI_API_KEY=your_api_key_here

# Model Configurations
WHISPER_MODEL=base  # tiny, base, small, medium, large
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_MODEL=gpt-3.5-turbo

# Chunking Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Retrieval Parameters
TOP_K=5
```

## Cấu trúc thư mục

```
CODE/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env.example           # Environment config template
├── README.md              # Hướng dẫn này
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Cấu hình hệ thống
│   └── modules/
│       ├── __init__.py
│       ├── asr_module.py         # Whisper ASR
│       ├── chunking_module.py    # Text chunking
│       ├── embedding_module.py   # Text embedding
│       ├── vector_db_module.py   # Vector database
│       └── rag_module.py         # RAG system
│
├── data/
│   ├── audio/             # Audio files input
│   ├── transcripts/       # ASR output
│   ├── vector_db/         # Vector database storage
│   └── outputs/           # Query results
│
└── outputs/               # Output files
```

## Sử dụng

### 1. Xử lý Audio Files

**Xử lý một file audio:**

```bash
python main.py --mode process --audio data/audio/sample.mp3
```

**Xử lý tất cả audio files trong thư mục:**

```bash
python main.py --mode process --audio data/audio/
```

Pipeline sẽ tự động:
1. Transcribe audio → văn bản + timestamps
2. Chunking văn bản theo ngữ nghĩa
3. Tạo embeddings
4. Lưu vào vector database

### 2. Truy vấn hệ thống

**Query một lần:**

```bash
python main.py --mode query --question "Nội dung chính của audio là gì?"
```

**Query với custom top_k:**

```bash
python main.py --mode query --question "Ai là diễn giả?" --top-k 3
```

### 3. Interactive Mode

```bash
python main.py --mode interactive
```

Chế độ tương tác cho phép:
- Đặt nhiều câu hỏi liên tục
- Xem thống kê database (`stats`)
- Thoát bằng `exit` hoặc `quit`

## Ví dụ sử dụng

### Example 1: Xử lý podcast

```bash
# Bước 1: Xử lý audio
python main.py --mode process --audio data/audio/podcast_episode1.mp3

# Bước 2: Query
python main.py --mode query --question "Chủ đề chính được thảo luận là gì?"
```

### Example 2: Batch processing

```python
from main import AudioIRPipeline

# Initialize
pipeline = AudioIRPipeline()

# Process multiple audios
audio_files = [
    "data/audio/lecture1.mp3",
    "data/audio/lecture2.mp3",
    "data/audio/lecture3.mp3"
]

results = pipeline.process_audio_batch(audio_files)

# Query
response = pipeline.query("Nội dung bài giảng về AI là gì?")
print(response["answer"])

# Print sources with timestamps
for source in response["sources"]:
    print(f"[{source['start_time_formatted']}] {source['text'][:100]}...")
```

### Example 3: Sử dụng từng module riêng lẻ

```python
from src.modules import WhisperASR, TextChunker, TextEmbedding

# ASR
asr = WhisperASR(model_name="base")
transcript = asr.transcribe_audio("audio.mp3")

# Chunking
chunker = TextChunker(chunk_size=500, method="semantic")
chunks = chunker.chunk_transcript(transcript)

# Embedding
embedder = TextEmbedding()
chunks_with_embeddings = embedder.encode_chunks(chunks)
```

## Modules

### 1. ASR Module (asr_module.py)

**Chức năng:**
- Chuyển đổi audio → text với Whisper
- Tạo timestamps cho từng segment
- Hỗ trợ batch processing

**Models hỗ trợ:**
- `tiny`: Nhanh nhất, độ chính xác thấp
- `base`: Cân bằng (recommended)
- `small`: Chính xác hơn
- `medium`: Rất chính xác
- `large`: Chính xác nhất, chậm nhất

### 2. Chunking Module (chunking_module.py)

**Phương pháp chunking:**
- `fixed`: Chia theo kích thước cố định
- `sentence`: Chia theo câu
- `semantic`: Chia theo ngữ nghĩa (recommended)

**Features:**
- Preserve timestamps từ ASR
- Configurable chunk size và overlap
- Metadata cho mỗi chunk

### 3. Embedding Module (embedding_module.py)

**Models hỗ trợ:**
- `paraphrase-multilingual-mpnet-base-v2`: Tiếng Việt + đa ngôn ngữ (recommended)
- `all-MiniLM-L6-v2`: Tiếng Anh, nhẹ và nhanh
- Custom models từ HuggingFace

**Features:**
- Batch encoding
- Similarity computation
- Normalize embeddings

### 4. Vector Database Module (vector_db_module.py)

**Databases hỗ trợ:**
- **ChromaDB**: Persistent, dễ sử dụng (recommended)
- **FAISS**: Nhanh, hiệu quả với datasets lớn

**Features:**
- Add/Search documents
- Metadata filtering (ChromaDB)
- Persistent storage

### 5. RAG Module (rag_module.py)

**Chức năng:**
- Retrieval-Augmented Generation
- Kết hợp retrieval với LLM
- Custom prompt templates

**LLM Models hỗ trợ:**
- GPT-3.5-turbo
- GPT-4
- Custom OpenAI-compatible models

## Performance Tips

### GPU Acceleration

```python
# Trong config.py hoặc .env
WHISPER_DEVICE = "cuda"  # Sử dụng GPU
```

### Tối ưu Chunking

```python
# Chunk size nhỏ: Chính xác hơn nhưng nhiều chunks
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# Chunk size lớn: Ít chunks hơn, context dài hơn
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
```

### Vector Database Selection

- **ChromaDB**: Tốt cho development, dễ debug
- **FAISS**: Tốt cho production, datasets lớn (>10K documents)

## Troubleshooting

### Lỗi: "OPENAI_API_KEY chưa được cấu hình"

→ Thêm API key vào file `.env`

### Lỗi: Out of Memory khi chạy Whisper

→ Sử dụng model nhỏ hơn: `WHISPER_MODEL=tiny` hoặc `base`

### Lỗi: ChromaDB không tìm thấy documents

→ Kiểm tra đã process audio chưa: `python main.py --mode process --audio <file>`

### Query trả về kết quả không liên quan

→ Tăng `TOP_K` hoặc cải thiện chunking method

## Roadmap

### Giai đoạn 1 (Tuần 1-8): ✅ Completed
- [x] ASR với Whisper
- [x] Text Chunking
- [x] Text Embedding
- [x] Vector Database
- [x] RAG với LLM
- [x] Basic CLI

### Giai đoạn 1 (Tuần 9-14): Cải thiện
- [ ] Fine-tuning retrieval
- [ ] Evaluation metrics
- [ ] REST API
- [ ] Web interface

### Giai đoạn 2: Multimodal (Future)
- [ ] Audio Embedding (Wav2Vec2/HuBERT)
- [ ] Early/Late Fusion
- [ ] Speaker Diarization
- [ ] Speech Emotion Recognition

## License

MIT License

## Liên hệ

- Author: [Your Name]
- Email: [your.email@example.com]
- GitHub: [your-github-profile]

---

**Developed as part of Đồ án chuyên nghành - 2025**
