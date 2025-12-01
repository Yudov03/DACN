"""
Cấu hình hệ thống - Configuration Module
Chứa các cấu hình cho toàn bộ hệ thống IR đa phương thức
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Cấu hình chính của hệ thống"""

    # ========================
    # Project Paths
    # ========================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    AUDIO_DIR = DATA_DIR / "audio"
    TRANSCRIPT_DIR = DATA_DIR / "transcripts"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    OUTPUT_DIR = BASE_DIR / "outputs"

    # ========================
    # API Keys
    # ========================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # ========================
    # Model Configurations
    # ========================
    # Whisper ASR Model
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
    WHISPER_DEVICE = "cuda"  # or "cpu"

    # Text Embedding Model
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    EMBEDDING_DIMENSION = 768  # Dimension cho model trên

    # LLM Model
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 500

    # ========================
    # Chunking Parameters
    # ========================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # số ký tự
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # số ký tự overlap
    CHUNKING_METHOD = "semantic"  # "fixed", "semantic", "sentence"

    # ========================
    # Vector Database Parameters
    # ========================
    VECTOR_DB_TYPE = "chromadb"  # "chromadb" or "faiss"
    COLLECTION_NAME = "audio_transcripts"

    # ========================
    # Retrieval Parameters
    # ========================
    TOP_K = int(os.getenv("TOP_K", "5"))  # Số lượng chunks retrieve
    SIMILARITY_THRESHOLD = 0.5  # Ngưỡng similarity score

    # ========================
    # RAG Parameters
    # ========================
    RAG_PROMPT_TEMPLATE = """Bạn là một trợ lý AI thông minh. Dựa trên các đoạn văn bản sau đây được trích xuất từ âm thanh, hãy trả lời câu hỏi của người dùng một cách chính xác và chi tiết.

Các đoạn văn bản liên quan:
{context}

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên thông tin được cung cấp. Nếu thông tin không đủ để trả lời, hãy nói rõ điều đó.

Trả lời:"""

    @classmethod
    def ensure_directories(cls):
        """Tạo các thư mục cần thiết nếu chưa tồn tại"""
        for directory in [cls.DATA_DIR, cls.AUDIO_DIR, cls.TRANSCRIPT_DIR,
                         cls.VECTOR_DB_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Kiểm tra cấu hình có hợp lệ không"""
        if not cls.OPENAI_API_KEY and cls.LLM_MODEL.startswith("gpt"):
            # Use ASCII-safe message for Windows console compatibility
            try:
                print("WARNING: OPENAI_API_KEY chua duoc cau hinh!")
            except UnicodeEncodeError:
                print("WARNING: OPENAI_API_KEY not configured!")

        cls.ensure_directories()
        return True


# Khởi tạo và validate config khi import
Config.validate()
