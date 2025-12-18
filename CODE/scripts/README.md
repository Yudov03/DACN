# Scripts Directory

Demo scripts and admin scripts for the Multimodal Information Retrieval system.

## Directory Structure

```
scripts/
├── import_resources.py         # ALL-IN-ONE: Process + Index + Embed
├── process_resources.py        # Step 1: Process files only (OCR/ASR)
├── reindex_documents.py        # Step 2: Post-process + Index
├── demo_rag_pipeline.py        # RAG pipeline demo
├── demo_document_processor.py  # Document processing demo
├── demo_anti_hallucination.py  # Anti-hallucination demo
├── demo_tts.py                 # Text-to-Speech demo
├── demo_knowledge_base.py      # Knowledge Base demo
└── README.md
```

## Admin Scripts - Document Import Workflow

### Option A: All-in-One (Recommended)

#### import_resources.py

Script chính để import tài liệu từ `data/resource/` vào Knowledge Base. Làm tất cả trong một lần chạy.

```bash
# Xem trước files sẽ import (không thực sự import)
python scripts/import_resources.py --dry-run

# Import tất cả vào Knowledge Base
python scripts/import_resources.py

# XÓA TOÀN BỘ KB và import lại từ đầu
python scripts/import_resources.py --clear
```

**Pipeline:** Resource → Copy → Process (OCR/ASR) → Post-process → Chunk → Embed → Qdrant

### Option B: Two-Step Workflow (Advanced)

Chia thành 2 bước để linh hoạt hơn khi cần re-index hoặc debug.

```
process_resources.py + reindex_documents.py = import_resources.py
```

#### Step 1: process_resources.py

Xử lý files từ `data/resource/` và lưu kết quả RAW vào `data/knowledge_base/`.

```bash
# Xem trước files sẽ xử lý
python scripts/process_resources.py --dry-run

# Xử lý tất cả files (OCR/ASR → RAW content)
python scripts/process_resources.py

# XÓA TOÀN BỘ processed data và xử lý lại
python scripts/process_resources.py --clear
```

**Pipeline:** Resource → Copy → Process (OCR/ASR) → Save RAW to processed/

**Output:**
- `data/knowledge_base/documents/` - Copy của files gốc
- `data/knowledge_base/transcripts/` - Kết quả ASR (audio/video)
- `data/knowledge_base/processed/` - JSON với nội dung RAW

#### Step 2: reindex_documents.py

Post-process và index vào Qdrant từ processed data.

```bash
# Index tất cả documents
python scripts/reindex_documents.py

# Reset Qdrant và index lại
python scripts/reindex_documents.py --reset

# Index một document cụ thể (tự động xóa chunks cũ)
python scripts/reindex_documents.py --file doc_98e6508e
```

**Pipeline:** processed/ → Post-process (LLM) → Chunk → Embed → Qdrant

**Features:**
- Post-processing với **caching** (Cache HIT ~0.0s)
- Tự động **delete old chunks** khi index lại từng file
- Progress tracking với timestamps

### Khi nào dùng Option nào?

| Scenario | Recommend |
|----------|-----------|
| Import lần đầu | `import_resources.py` |
| Re-import sau khi thêm files | `import_resources.py` |
| Thay đổi embedding model | `reindex_documents.py --reset` |
| Thay đổi chunking config | `reindex_documents.py --reset` |
| Thay đổi post-processing | `reindex_documents.py --reset` |
| Debug từng bước | `process_resources.py` → `reindex_documents.py` |
| Test ASR/OCR | `process_resources.py` |

## Demo Scripts

### 1. demo_rag_pipeline.py
Complete RAG pipeline demonstration.

**Features:**
- Text chunking (fixed, sentence, recursive)
- Embedding (local SBERT/E5, Google, OpenAI)
- Vector database (Qdrant + BM25 hybrid)
- Semantic search
- RAG query with LLM

```bash
# Basic demo
python scripts/demo_rag_pipeline.py

# With specific embedding provider
python scripts/demo_rag_pipeline.py --provider local
python scripts/demo_rag_pipeline.py --provider google

# Skip RAG query (if no LLM available)
python scripts/demo_rag_pipeline.py --skip-rag
```

### 2. demo_document_processor.py
Document processing with 68 supported formats.

**Features:**
- UnifiedProcessor auto-detection
- Text, PDF, Word, Excel processing
- OCR for images and scanned PDFs
- ASR for audio/video files
- Batch processing

```bash
# Basic demo
python scripts/demo_document_processor.py

# Process specific file
python scripts/demo_document_processor.py --file data/documents/report.pdf

# Batch process folder
python scripts/demo_document_processor.py --folder data/documents/
```

### 3. demo_anti_hallucination.py
Anti-hallucination modules demonstration.

**Features:**
- Answer Verification (grounding check)
- Abstention Checker (know when to say "I don't know")
- Conflict Detection (handle conflicting information)
- Integrated flow

```bash
python scripts/demo_anti_hallucination.py
```

### 4. demo_tts.py
Text-to-Speech demonstration.

**Features:**
- Voice options (Vietnamese, English)
- Speech synthesis
- Rate and volume control
- Save to file

```bash
# Basic demo
python scripts/demo_tts.py

# Custom text
python scripts/demo_tts.py --text "Xin chào" --voice vi-female

# Save to file
python scripts/demo_tts.py --text "Hello" --voice en-male --save output.mp3
```

### 5. demo_knowledge_base.py
Knowledge Base management demonstration.

**Features:**
- Add/remove documents
- Search by filename, tags
- Semantic search
- Export/Import backup

```bash
# Basic demo
python scripts/demo_knowledge_base.py

# With real documents
python scripts/demo_knowledge_base.py --folder data/documents/
```

## Quick Demo Commands

```bash
# Import dữ liệu (chạy trước)
python scripts/import_resources.py

# Demo system
python scripts/demo_rag_pipeline.py --provider local
python scripts/demo_document_processor.py
python scripts/demo_anti_hallucination.py
python scripts/demo_tts.py
python scripts/demo_knowledge_base.py

# Presentation demo (key features)
python scripts/demo_rag_pipeline.py --provider local
python scripts/demo_anti_hallucination.py
python scripts/demo_tts.py --text "Hệ thống hoạt động tốt" --voice vi-female
```

## Demo Output Examples

### import_resources.py
```
[09:15:32] ================================================================
[09:15:32]   IMPORT RESOURCES TO KNOWLEDGE BASE
[09:15:32] ================================================================
[09:15:32] Source: C:\...\data\resource
[09:15:32] Target: C:\...\data\knowledge_base
[09:15:32] ================================================================
[09:15:32] Found 3 files to process
[09:15:32] ----------------------------------------------------------------
[09:15:32] [1/3] Processing: QD_2349.pdf
[09:15:33]   + Processed: 9 pages, 40 chunks
[09:15:34]   + Indexed to Qdrant
[09:15:34] ----------------------------------------------------------------
[09:15:34] [2/3] Processing: lecture.mp3
[09:15:35]   + ASR: 4.28s audio → 2 chunks
[09:15:35]   + Post-processing (Cache HIT - 0.0s)
[09:15:36]   + Indexed to Qdrant
[09:15:36] ================================================================
[09:15:36] COMPLETED: 3 files, 43 chunks, 4.2s total
```

### reindex_documents.py (single file)
```
[10:22:15] ================================================================
[10:22:15]   REINDEX: doc_6e43c6d5 (lecture.mp3)
[10:22:15] ================================================================
[10:22:15]   Reading processed data...
[10:22:15]   Post-processing: Cache HIT (0.0s)
[10:22:15]   Chunking: 2 chunks
[10:22:15]   Embedding...
[10:22:16]   Deleted old chunks for doc_6e43c6d5
[10:22:16]   Uploaded 2 chunks to Qdrant
[10:22:16] ================================================================
[10:22:16] COMPLETED: 1 file, 2 chunks, 1.1s
```

### Anti-Hallucination Demo
```
======================================
  1. ANSWER VERIFICATION
======================================
Context: Học phí năm 2024 là 15 triệu đồng...
Question: Học phí một kỳ là bao nhiêu?
Answer: Học phí là 15 triệu đồng mỗi kỳ.

Result:
  Grounding: FULLY_GROUNDED
  Confidence: 0.95
```

## Notes

- All scripts include Windows encoding fix for Vietnamese
- Scripts use local embedding by default (no API key required)
- TTS uses edge-tts (free, no API key)
- Post-processing results are **cached** (MD5 hash key)
- Temporary files are automatically cleaned up
