# Data Directory

Thư mục chứa dữ liệu runtime của hệ thống RAG.

## Cấu trúc

```
data/
├── resource/                  # INPUT - Nhà trường upload tài liệu vào đây
│   ├── documents/             # PDFs, Word, Excel, Images, Code...
│   └── audio/                 # Audio (MP3, WAV) và Video (MP4, AVI)
│
├── knowledge_base/            # PROCESSED - Hệ thống xử lý và lưu trữ
│   ├── index.json             # Registry metadata của tất cả documents
│   ├── documents/             # Copy của documents đã import
│   │   ├── pdf/               # PDF files
│   │   ├── audio/             # Audio files
│   │   └── video/             # Video files
│   ├── transcripts/           # Kết quả ASR từ audio/video (.txt)
│   └── processed/             # Processed JSON với nội dung RAW
│
└── cache/                     # CACHE - Tăng tốc post-processing
    └── post_processing/
        ├── index.json         # Cache index (MD5 hash → metadata)
        └── chunks/            # Cached post-processed content
```

## Luồng dữ liệu

```
┌─────────────────────────────────────────────────────────────┐
│  ADMIN (Nhà trường)                                         │
│                                                             │
│  1. Upload files vào data/resource/                         │
│     - documents/ : PDF, DOCX, XLSX, Images...               │
│     - audio/     : MP3, WAV, MP4, AVI...                    │
│                                                             │
│  2. Chạy lệnh import (chọn 1 trong 2 cách):                 │
│                                                             │
│     [ALL-IN-ONE]                                            │
│     python scripts/import_resources.py                      │
│                                                             │
│     [TWO-STEP]                                              │
│     python scripts/process_resources.py                     │
│     python scripts/reindex_documents.py                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  HỆ THỐNG XỬ LÝ TỰ ĐỘNG                                     │
│                                                             │
│  Step 1: Process (OCR/ASR)                                  │
│  - Documents → Extract text                                 │
│  - Audio/Video → Whisper ASR → Transcript                   │
│  - Images/PDF scan → OCR                                    │
│  → Lưu RAW vào knowledge_base/processed/                    │
│                                                             │
│  Step 2: Index (Post-process + Embed)                       │
│  - RAW → Post-processing (LLM) → Cleaned text               │
│  - Cleaned → Chunk → Embed → Qdrant                         │
│  → Cache post-processing results                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  USER (Sinh viên)                                           │
│                                                             │
│  - Tìm kiếm thông tin qua Web UI hoặc CLI                   │
│  - Hệ thống trả lời dựa trên knowledge_base                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Commands

### Admin - Import tài liệu

#### Option A: All-in-One (Recommended)

```bash
# Xem trước files sẽ import
python scripts/import_resources.py --dry-run

# Import tất cả vào Knowledge Base
python scripts/import_resources.py

# Xóa KB cũ và import lại từ đầu (reset hoàn toàn)
python scripts/import_resources.py --clear
```

**Pipeline:** resource/ → Copy → Process → Post-process → Chunk → Embed → Qdrant

#### Option B: Two-Step (Advanced)

```bash
# Step 1: Process files (OCR/ASR) - lưu RAW content
python scripts/process_resources.py

# Step 2: Post-process và index vào Qdrant
python scripts/reindex_documents.py
```

**Khi nào dùng Two-Step?**
- Thay đổi embedding model → chỉ cần chạy lại `reindex_documents.py --reset`
- Thay đổi chunking config → chỉ cần chạy lại `reindex_documents.py --reset`
- Debug OCR/ASR → chạy `process_resources.py` rồi kiểm tra processed/
- Re-index một file → `reindex_documents.py --file doc_id`

### Clear commands

| Command | Xóa gì |
|---------|--------|
| `import_resources.py --clear` | Toàn bộ KB + Qdrant, import lại từ resource/ |
| `process_resources.py --clear` | Xóa documents/, processed/, transcripts/, xử lý lại từ resource/ |
| `reindex_documents.py --reset` | Xóa collection Qdrant, index lại từ processed/ |

**Lưu ý:** `--clear` xóa FILES trong thư mục, không xóa thư mục.

### User - Tra cứu

```bash
# Student Portal - Web UI cho sinh viên (auto-start Ollama)
streamlit run app.py
# http://localhost:8501

# Admin Portal - Web UI cho quản trị viên
streamlit run app_admin.py --server.port 8502
# http://localhost:8502

# CLI interactive mode
python main.py --mode interactive
```

### Kiểm tra dữ liệu

```bash
# Xem thống kê Knowledge Base
python -c "
from src.modules import KnowledgeBase
kb = KnowledgeBase(base_dir='data/knowledge_base')
stats = kb.get_stats()
print(f'Documents: {stats.total_documents}')
print(f'Chunks: {stats.total_chunks}')
print(f'Size: {stats.total_size_mb:.2f} MB')
"

# Xem cache post-processing
python -c "
import json
with open('data/cache/post_processing/index.json') as f:
    cache = json.load(f)
print(f'Cached entries: {len(cache)}')
"
```

## Cache System

Post-processing (LLM) rất chậm (~5-30s/chunk). Hệ thống cache kết quả để tăng tốc khi re-index.

### Cơ chế hoạt động

1. **Hash key**: MD5 của `{method}:{model}:{content}`
2. **Cache HIT**: Lấy kết quả từ file, ~0.0s
3. **Cache MISS**: Gọi LLM, ~5-30s, lưu vào cache

### Cache structure

```
data/cache/post_processing/
├── index.json                    # Metadata index
│   {
│     "abc123...": {
│       "method": "ollama",
│       "model": "qwen2.5:7b",
│       "input_length": 140,
│       "output_length": 153,
│       "created_at": "2025-12-19T04:22:49"
│     }
│   }
└── chunks/
    └── abc123....txt            # Cached content
```

### Khi nào cache bị invalidate?

Cache tự động invalidate khi:
- Thay đổi post-processing method (env: `POST_PROCESSING_METHOD`)
- Thay đổi LLM model (env: `OLLAMA_MODEL`)
- Nội dung RAW khác (file khác hoặc re-process)

## Supported Formats (68 extensions)

| Category | Extensions |
|----------|------------|
| **Documents** | .pdf, .docx, .doc |
| **Presentations** | .pptx, .ppt |
| **Spreadsheets** | .xlsx, .xls |
| **Text/Data** | .txt, .md, .csv, .tsv, .json, .xml, .html, .log, .ini, .cfg, .rtf |
| **Code** | .py, .js, .ts, .jsx, .tsx, .java, .kt, .cpp, .c, .h, .hpp, .go, .rs, .rb, .php, .swift, .cs, .vb, .sql, .sh, .bash, .ps1, .yaml, .yml, .toml, .r, .R, .scala |
| **Images (OCR)** | .png, .jpg, .jpeg, .bmp, .tiff, .tif, .webp |
| **Audio** | .mp3, .wav, .m4a, .flac, .ogg, .wma, .aac |
| **Video** | .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v |
