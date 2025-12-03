"""
Cau hinh he thong - Configuration Module
Chua cac cau hinh cho toan bo he thong IR da phuong thuc
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Cau hinh chinh cua he thong"""

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
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

    # ========================
    # Provider Selection (openai or google)
    # ========================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")  # openai or google
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")  # openai or google

    # ========================
    # Qdrant Configuration
    # ========================
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_URL = os.getenv("QDRANT_URL", None)  # For Qdrant Cloud
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # For Qdrant Cloud
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "audio_transcripts")

    # ========================
    # Model Configurations
    # ========================
    # Whisper ASR Model
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # or "cpu"

    # OpenAI Models
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_EMBEDDING_DIMENSION = int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536"))
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    # Google Models
    GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
    GOOGLE_EMBEDDING_DIMENSION = int(os.getenv("GOOGLE_EMBEDDING_DIMENSION", "768"))
    GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.0-flash")

    # Active model (based on provider)
    @classmethod
    def get_embedding_model(cls):
        if cls.EMBEDDING_PROVIDER == "google":
            return cls.GOOGLE_EMBEDDING_MODEL
        return cls.OPENAI_EMBEDDING_MODEL

    @classmethod
    def get_embedding_dimension(cls):
        if cls.EMBEDDING_PROVIDER == "google":
            return cls.GOOGLE_EMBEDDING_DIMENSION
        return cls.OPENAI_EMBEDDING_DIMENSION

    @classmethod
    def get_llm_model(cls):
        if cls.LLM_PROVIDER == "google":
            return cls.GOOGLE_LLM_MODEL
        return cls.OPENAI_LLM_MODEL

    @classmethod
    def get_api_key(cls, provider=None):
        """Get API key for specified or default provider"""
        if provider == "google" or (provider is None and cls.LLM_PROVIDER == "google"):
            return cls.GOOGLE_API_KEY
        return cls.OPENAI_API_KEY

    # Legacy support
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))  # Default to Google dimension
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")  # Default to Google model
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # ========================
    # Chunking Parameters
    # ========================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # so ky tu
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # so ky tu overlap
    CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "semantic")  # "fixed", "semantic", "sentence"

    # Semantic chunking parameters
    SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))
    SEMANTIC_WINDOW_SIZE = int(os.getenv("SEMANTIC_WINDOW_SIZE", "5"))

    # ========================
    # Retrieval Parameters
    # ========================
    TOP_K = int(os.getenv("TOP_K", "5"))  # So luong chunks retrieve
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))  # Nguong similarity score

    # MMR (Maximal Marginal Relevance) parameters
    MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # Balance between relevance and diversity

    # ========================
    # RAG Parameters
    # ========================
    RAG_PROMPT_TEMPLATE = """Ban la mot tro ly AI thong minh. Dua tren cac doan van ban sau day duoc trich xuat tu am thanh, hay tra loi cau hoi cua nguoi dung mot cach chinh xac va chi tiet.

Cac doan van ban lien quan (kem timestamp):
{context}

Cau hoi: {question}

Yeu cau:
1. Tra loi dua tren thong tin duoc cung cap
2. Neu thong tin khong du de tra loi, hay noi ro "Toi khong tim thay thong tin lien quan"
3. Trich dan nguon voi format [file_id:start_time-end_time] khi co the
4. Tra loi ngan gon, chinh xac

Tra loi:"""

    @classmethod
    def ensure_directories(cls):
        """Tao cac thu muc can thiet neu chua ton tai"""
        for directory in [cls.DATA_DIR, cls.AUDIO_DIR, cls.TRANSCRIPT_DIR,
                         cls.VECTOR_DB_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Kiem tra cau hinh co hop le khong"""
        # Check API key based on provider
        if cls.LLM_PROVIDER == "google" or cls.EMBEDDING_PROVIDER == "google":
            if not cls.GOOGLE_API_KEY:
                print("WARNING: GOOGLE_API_KEY chua duoc cau hinh!")
        if cls.LLM_PROVIDER == "openai" or cls.EMBEDDING_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                print("WARNING: OPENAI_API_KEY chua duoc cau hinh!")

        cls.ensure_directories()
        return True

    @classmethod
    def get_qdrant_config(cls):
        """Lay cau hinh Qdrant"""
        if cls.QDRANT_URL:
            # Qdrant Cloud
            return {
                "url": cls.QDRANT_URL,
                "api_key": cls.QDRANT_API_KEY
            }
        else:
            # Local Qdrant
            return {
                "host": cls.QDRANT_HOST,
                "port": cls.QDRANT_PORT
            }


# Khoi tao va validate config khi import
Config.validate()
