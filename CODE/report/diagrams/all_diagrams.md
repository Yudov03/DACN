# Sơ đồ kỹ thuật cho Báo cáo ĐACN

Các sơ đồ dưới đây có thể render bằng Mermaid Live Editor (https://mermaid.live) hoặc các công cụ hỗ trợ Mermaid.

---

## 1. Quy trình ASR truyền thống (Chapter 2 - fig:asr_pipeline)

```mermaid
flowchart LR
    A[Tín hiệu<br/>âm thanh] --> B[Tiền xử lý<br/>tín hiệu]
    B --> C[Trích xuất<br/>đặc trưng<br/>MFCC]
    C --> D[Mô hình<br/>âm học]
    D --> E[Mô hình<br/>ngôn ngữ]
    E --> F[Giải mã<br/>Decoding]
    F --> G[Văn bản]

    style A fill:#e1f5fe
    style G fill:#c8e6c9
```

---

## 2. Kiến trúc Transformer (Chapter 2 - fig:transformer_architectures)

```mermaid
flowchart TB
    subgraph BERT["BERT (Encoder-only)"]
        direction TB
        B1[Input] --> B2[Bidirectional<br/>Encoder]
        B2 --> B3[Classification<br/>NER, QA]
    end

    subgraph GPT["GPT (Decoder-only)"]
        direction TB
        G1[Input] --> G2[Left-to-Right<br/>Decoder]
        G2 --> G3[Text Generation]
    end

    subgraph T5["T5 (Encoder-Decoder)"]
        direction TB
        T1[Input] --> T2[Encoder]
        T2 --> T3[Decoder]
        T3 --> T4[Seq2Seq Tasks]
    end

    style BERT fill:#e3f2fd
    style GPT fill:#fce4ec
    style T5 fill:#f3e5f5
```

---

## 3. Kiến trúc RAG cơ bản (Chapter 2 - fig:rag_architecture)

```mermaid
flowchart LR
    subgraph Indexing["Giai đoạn Indexing (Offline)"]
        D1[Documents] --> D2[Chunking]
        D2 --> D3[Embedding]
        D3 --> D4[(Vector DB)]
    end

    subgraph Query["Giai đoạn Query (Online)"]
        Q1[Câu hỏi] --> Q2[Query<br/>Embedding]
        Q2 --> Q3[Vector<br/>Search]
        D4 --> Q3
        Q3 --> Q4[Top-K<br/>Documents]
        Q4 --> Q5[Prompt<br/>Construction]
        Q5 --> Q6[LLM]
        Q6 --> Q7[Câu trả lời<br/>có nguồn]
    end

    style D1 fill:#fff3e0
    style Q1 fill:#e1f5fe
    style Q7 fill:#c8e6c9
```

---

## 4. Không gian Embedding (Chapter 2 - fig:embedding_space)

```mermaid
flowchart TB
    subgraph Space["Không gian Vector 2D"]
        subgraph Cluster1["Cụm: Người học"]
            S1((Sinh viên))
            S2((Học sinh))
            S3((Học viên))
        end

        subgraph Cluster2["Cụm: Phương tiện"]
            V1((Ô tô))
            V2((Xe máy))
            V3((Xe đạp))
        end

        subgraph Cluster3["Cụm: Giảng dạy"]
            T1((Giáo viên))
            T2((Giảng viên))
        end
    end

    style Cluster1 fill:#e3f2fd
    style Cluster2 fill:#fce4ec
    style Cluster3 fill:#e8f5e9
```

---

## 5. So sánh phương pháp Chunking (Chapter 2 - fig:chunking_methods)

```mermaid
flowchart TB
    subgraph Original["Văn bản gốc"]
        O1[Đoạn văn dài về nhiều chủ đề A, B, C...]
    end

    subgraph Fixed["Fixed-size Chunking"]
        F1[Chunk 1<br/>500 chars]
        F2[Chunk 2<br/>500 chars]
        F3[Chunk 3<br/>500 chars]
    end

    subgraph Semantic["Semantic Chunking"]
        S1[Chunk A<br/>Chủ đề A<br/>~400 chars]
        S2[Chunk B<br/>Chủ đề B<br/>~600 chars]
        S3[Chunk C<br/>Chủ đề C<br/>~350 chars]
    end

    Original --> Fixed
    Original --> Semantic

    style Fixed fill:#ffebee
    style Semantic fill:#e8f5e9
```

---

## 6. Kiến trúc tổng thể hệ thống (Chapter 4 - fig:system_architecture)

```mermaid
flowchart TB
    subgraph Presentation["Presentation Layer"]
        P1[Student Portal<br/>Streamlit]
        P2[Admin Portal<br/>Streamlit]
        P3[CLI]
    end

    subgraph Business["Business Logic Layer"]
        B1[ASR Module<br/>Faster-Whisper]
        B2[Document Processor<br/>OCR/PDF/Office]
        B3[RAG System<br/>Query + Generation]
        B4[Anti-Hallucination<br/>Verification]
        B5[Voice Input]
        B6[TTS Module<br/>Edge-TTS]
    end

    subgraph Data["Data Layer"]
        D1[(Qdrant<br/>Vector DB)]
        D2[(File System<br/>Knowledge Base)]
        D3[(Cache<br/>Post-processing)]
    end

    Presentation --> Business
    Business --> Data

    style Presentation fill:#e3f2fd
    style Business fill:#fff3e0
    style Data fill:#f3e5f5
```

---

## 7. Pipeline Processing - Bước 1 (Chapter 4 - fig:processing_pipeline)

```mermaid
flowchart LR
    A[Input Files<br/>data/resource/] --> B{Phát hiện<br/>loại file}

    B -->|Audio/Video| C[ASR Module<br/>Faster-Whisper]
    B -->|PDF/Image| D[OCR Engine<br/>PaddleOCR]
    B -->|Text/Office| E[Text Extractor<br/>PyMuPDF, docx]

    C --> F[RAW Content]
    D --> F
    E --> F

    F --> G[processed/*.json]

    style A fill:#e1f5fe
    style G fill:#c8e6c9
```

---

## 8. Pipeline Indexing - Bước 2 (Chapter 4 - fig:indexing_pipeline)

```mermaid
flowchart LR
    A[RAW Content<br/>processed/*.json] --> B[Post-Process<br/>LLM Correction]
    B --> C{Cache<br/>MD5 Hash}
    C -->|Hit| D[Load from Cache]
    C -->|Miss| E[Process & Save]
    D --> F[Chunking]
    E --> F
    F --> G[Embedding<br/>SBERT/E5]
    G --> H[(Qdrant<br/>Upload)]

    style A fill:#fff3e0
    style H fill:#c8e6c9
```

---

## 9. Query Pipeline (Chapter 4 - fig:query_pipeline)

```mermaid
flowchart LR
    subgraph Input
        I1[Text Input]
        I2[Voice Input 🎤]
    end

    I2 --> ASR[ASR]
    ASR --> Q
    I1 --> Q[Query]

    Q --> QE[Query<br/>Expansion]
    QE --> HS[Hybrid Search<br/>Vector + BM25]
    HS --> RR[Reranking]
    RR --> CD[Conflict<br/>Detection]
    CD --> RAG[RAG<br/>Generation]
    RAG --> AV[Answer<br/>Verification]
    AV --> ANS[Final Answer]
    ANS --> TTS[TTS 🔊]

    style I2 fill:#e1f5fe
    style ANS fill:#c8e6c9
    style TTS fill:#fff3e0
```

---

## 10. ASR Module Class Diagram (Chapter 4 - fig:asr_class)

```mermaid
classDiagram
    class WhisperASR {
        -model: WhisperModel
        -vad: SileroVAD
        -device: str
        -compute_type: str
        +transcribe(audio_path) TranscriptResult
        +transcribe_with_vad(audio_path) TranscriptResult
        -_load_audio(path) ndarray
        -_detect_voice_segments(audio) List
        -_merge_segments(segments) TranscriptResult
    }

    class TranscriptResult {
        +segments: List~TranscriptSegment~
        +language: str
        +duration: float
        +full_text: str
    }

    class TranscriptSegment {
        +start: float
        +end: float
        +text: str
        +confidence: float
        +words: List~WordInfo~
    }

    WhisperASR --> TranscriptResult
    TranscriptResult --> TranscriptSegment
```

---

## 11. Anti-Hallucination Architecture (Chapter 4 - fig:anti_hallucination)

```mermaid
flowchart LR
    A[Generated<br/>Answer] --> B[Answer<br/>Verifier]
    B --> C[Conflict<br/>Detector]
    C --> D[Abstention<br/>Checker]
    D --> E{Decision}

    E -->|Grounded| F[Verified Answer<br/>✓ High Confidence]
    E -->|Uncertain| G[Answer with<br/>Warning ⚠️]
    E -->|Abstain| H[Refuse to Answer<br/>❌ Suggest Alternatives]

    style A fill:#fff3e0
    style F fill:#c8e6c9
    style G fill:#fff9c4
    style H fill:#ffcdd2
```

---

## 12. Chunking Module Class Diagram (Chapter 4 - fig:chunking_class)

```mermaid
classDiagram
    class BaseChunker {
        <<abstract>>
        +chunk_size: int
        +chunk_overlap: int
        +chunk(text) List~Chunk~
    }

    class FixedChunker {
        +chunk(text) List~Chunk~
    }

    class RecursiveChunker {
        +separators: List~str~
        +chunk(text) List~Chunk~
    }

    class SemanticChunker {
        +embedder: TextEmbedding
        +threshold: float
        +chunk(text) List~Chunk~
        -_find_breakpoints(embeddings) List
    }

    class Chunk {
        +content: str
        +start_index: int
        +end_index: int
        +metadata: dict
    }

    BaseChunker <|-- FixedChunker
    BaseChunker <|-- RecursiveChunker
    BaseChunker <|-- SemanticChunker
    BaseChunker --> Chunk
```

---

## 13. Qdrant Collection Schema (Chapter 4 - fig:qdrant_schema)

```mermaid
classDiagram
    class QdrantCollection {
        +name: knowledge_base
        +vector_config: VectorConfig
        +payload_schema: PayloadSchema
    }

    class VectorConfig {
        +size: 768
        +distance: Cosine
        +on_disk: false
    }

    class PayloadSchema {
        +doc_id: string
        +source: string
        +file_type: string
        +chunk_index: int
        +start_time: float
        +end_time: float
        +created_at: datetime
        +content: string
    }

    QdrantCollection --> VectorConfig
    QdrantCollection --> PayloadSchema
```

---

## 14. Student Portal Layout (Chapter 4 - fig:student_wireframe)

```mermaid
flowchart TB
    subgraph Header["Header"]
        H1[Logo]
        H2[Tiêu đề: Student Portal]
        H3[Settings ⚙️]
    end

    subgraph Main["Main Layout"]
        subgraph Sidebar["Sidebar"]
            S1[Chat History]
            S2[Document List]
            S3[Auto-TTS Toggle]
        end

        subgraph Chat["Chat Area"]
            C1[Message Bubbles]
            C2[Source Citations]
            C3[Audio Player]
        end

        subgraph Input["Input Bar"]
            I1[Text Input]
            I2[Send Button]
            I3[Voice Button 🎤]
        end
    end

    Header --> Main
    Sidebar --- Chat
    Chat --> Input
```

---

## 15. Admin Portal Layout (Chapter 4 - fig:admin_wireframe)

```mermaid
flowchart TB
    subgraph Header["Admin Header"]
        H1[Logo]
        H2[Admin Portal]
        H3[User Info]
    end

    subgraph Tabs["Tab Navigation"]
        T1[Documents]
        T2[Upload]
        T3[Statistics]
        T4[Settings]
    end

    subgraph Content["Content Area"]
        subgraph DocTab["Documents Tab"]
            D1[Document Table]
            D2[Search/Filter]
            D3[Actions: View/Delete/Re-index]
        end

        subgraph UploadTab["Upload Tab"]
            U1[Drag & Drop Zone]
            U2[File List]
            U3[Import Button]
        end

        subgraph StatsTab["Statistics Tab"]
            ST1[Total Documents]
            ST2[Charts by Type]
            ST3[Recent Activity]
        end
    end

    Header --> Tabs
    Tabs --> Content
```

---

## 16. Two-Step Pipeline Overview (Bonus)

```mermaid
flowchart TB
    subgraph Step1["Bước 1: Processing (process_resources.py)"]
        direction LR
        A1[data/resource/] --> A2[Copy Files]
        A2 --> A3[OCR/ASR]
        A3 --> A4[processed/*.json]
    end

    subgraph Step2["Bước 2: Indexing (reindex_documents.py)"]
        direction LR
        B1[processed/*.json] --> B2[Post-process<br/>+ Cache]
        B2 --> B3[Chunking]
        B3 --> B4[Embedding]
        B4 --> B5[(Qdrant)]
    end

    Step1 --> Step2

    style Step1 fill:#e3f2fd
    style Step2 fill:#e8f5e9
```

---

## 17. Tech Stack Layers (Chapter 3 - fig:tech_stack_layers)

```mermaid
flowchart TB
    subgraph Layer3["Layer 3: Application"]
        direction LR
        A1[Streamlit<br/>Web UI]
        A2[Edge-TTS<br/>Voice Output]
        A3[Audio Recorder<br/>Voice Input]
    end

    subgraph Layer2["Layer 2: Framework & Orchestration"]
        direction LR
        F1[Ollama<br/>LLM Runtime]
        F2[Qdrant<br/>Vector DB]
        F3[sentence-transformers<br/>Embedding]
        F4[Pydantic<br/>Validation]
    end

    subgraph Layer1["Layer 1: Core Models"]
        direction LR
        M1[Faster-Whisper<br/>ASR]
        M2[Qwen2.5:7B<br/>LLM]
        M3[SBERT/E5<br/>Embedding]
        M4[PaddleOCR<br/>OCR]
    end

    Layer3 --> Layer2
    Layer2 --> Layer1

    style Layer3 fill:#e8f5e9
    style Layer2 fill:#fff3e0
    style Layer1 fill:#e3f2fd
```

---

## Hướng dẫn sử dụng

1. **Mermaid Live Editor**: https://mermaid.live
   - Copy paste code Mermaid vào editor
   - Export PNG/SVG

2. **VS Code Extension**: "Mermaid Markdown Syntax Highlighting"
   - Preview trực tiếp trong VS Code

3. **Draw.io**: https://draw.io
   - Import Mermaid code hoặc vẽ lại đẹp hơn

4. **Kích thước khuyến nghị**:
   - Sơ đồ ngang: width 0.9\textwidth
   - Sơ đồ dọc: width 0.7-0.8\textwidth
   - Class diagram: width 0.85\textwidth


### Objective dataflow
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '12px', 'fontFamily': 'Arial'}}}%%
flowchart TB
    %% ========== ROW 1: IMPORT PIPELINE ==========
    subgraph ROW1["📥 IMPORT PIPELINE"]
        direction LR
        A1["🎵 Audio/Video"] --> B1["ASR+VAD"]
        A2["📄 Documents"] --> B2["OCR"]
        B1 --> B3["Post-Process"]
        B2 --> B3
        B3 --> C1["Chunking"] --> C2["Embedding"] --> DB[("💾 VectorDB")]
    end

    %% ========== ROW 2: QUERY PIPELINE ==========
    subgraph ROW2["🔍 QUERY PIPELINE"]
        direction LR
        Q1["🎤 Query"] --> D1["Hybrid Search"] --> D2["Rerank"] --> D3["LLM"]
        D3 --> E1["Verify"] --> E2["Resolve"] --> E3["Abstain"] --> F1["🔊 Response"]
    end

    %% ========== CROSS-ROW CONNECTION ==========
    DB -.->|retrieval| D1

    %% ========== STYLING ==========
    style ROW1 fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style ROW2 fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style DB fill:#FFE0B2,stroke:#FF9800,stroke-width:2px

    style A1 fill:#C8E6C9,stroke:#388E3C
    style A2 fill:#C8E6C9,stroke:#388E3C
    style B1 fill:#BBDEFB,stroke:#1976D2
    style B2 fill:#BBDEFB,stroke:#1976D2
    style B3 fill:#BBDEFB,stroke:#1976D2
    style C1 fill:#FFF59D,stroke:#FBC02D
    style C2 fill:#FFF59D,stroke:#FBC02D

    style Q1 fill:#B2EBF2,stroke:#0097A7
    style D1 fill:#E1BEE7,stroke:#7B1FA2
    style D2 fill:#E1BEE7,stroke:#7B1FA2
    style D3 fill:#E1BEE7,stroke:#7B1FA2
    style E1 fill:#FFCDD2,stroke:#D32F2F
    style E2 fill:#FFCDD2,stroke:#D32F2F
    style E3 fill:#FFCDD2,stroke:#D32F2F
    style F1 fill:#A5D6A7,stroke:#388E3C
