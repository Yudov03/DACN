"""
Embedding Module - Tao vector embeddings cho van ban
Ho tro nhieu providers:
- Local: Sentence-BERT, E5 (sentence-transformers)
- Cloud: OpenAI, Google (LangChain)
"""

import os
import hashlib
import json
from typing import List, Dict, Union, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time


# Supported local models with their dimensions
LOCAL_MODELS = {
    # Sentence-BERT models
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    # E5 models
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "intfloat/e5-small-v2": 384,
    # Vietnamese-specific models
    "keepitreal/vietnamese-sbert": 768,
    "bkai-foundation-models/vietnamese-bi-encoder": 768,
}

# Short aliases for convenience
MODEL_ALIASES = {
    # Sentence-BERT
    "sbert": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sbert-mini": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    # E5
    "e5": "intfloat/multilingual-e5-base",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-large": "intfloat/multilingual-e5-large",
    "e5-small": "intfloat/multilingual-e5-small",
    # Vietnamese
    "vi-sbert": "keepitreal/vietnamese-sbert",
    "vi-encoder": "bkai-foundation-models/vietnamese-bi-encoder",
}


class TextEmbedding:
    """
    Lop tao embeddings cho van ban - ho tro nhieu providers:
    - Local (sentence-transformers): Sentence-BERT, E5 - khong can API key
    - OpenAI: text-embedding-3-small (1536d), text-embedding-3-large (3072d)
    - Google: models/text-embedding-004 (768d)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: str = "local",  # "local", "openai", or "google"
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None  # "cuda", "cpu", or None (auto)
    ):
        """
        Khoi tao Text Embedding model

        Args:
            model_name: Ten model (neu None se dung default theo provider)
                - Local: "sbert", "e5", "e5-large", hoac full HuggingFace path
                - OpenAI: "text-embedding-3-small", "text-embedding-3-large"
                - Google: "models/text-embedding-004"
            provider: "local" (Sentence-BERT/E5), "openai", hoac "google"
            api_key: API key (chi can cho openai/google)
            cache_dir: Thu muc cache embeddings
            device: Device cho local models ("cuda", "cpu", None=auto)
        """
        self.provider = provider.lower()
        self.device = device

        # Resolve model alias
        if model_name and model_name in MODEL_ALIASES:
            model_name = MODEL_ALIASES[model_name]

        # Default models
        if model_name is None:
            if self.provider == "local":
                model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            elif self.provider == "google":
                model_name = "models/text-embedding-004"
            else:
                model_name = "text-embedding-3-small"

        self.model_name = model_name

        # Get API key (only needed for cloud providers)
        if self.provider in ["openai", "google"]:
            if api_key:
                self.api_key = api_key
            elif self.provider == "google":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            else:
                self.api_key = os.getenv("OPENAI_API_KEY")

            if not self.api_key:
                key_name = "GOOGLE_API_KEY" if self.provider == "google" else "OPENAI_API_KEY"
                raise ValueError(f"{key_name} chua duoc cau hinh!")
        else:
            self.api_key = None

        # Initialize embeddings based on provider
        self._init_embeddings()

        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def _init_embeddings(self):
        """Initialize embeddings model based on provider"""
        print(f"Dang khoi tao {self.provider.upper()} Embedding model '{self.model_name}'...")

        if self.provider == "local":
            self._init_local_embeddings()

        elif self.provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=self.api_key
            )
            # Google embedding dimensions
            if "text-embedding-004" in self.model_name:
                self.embedding_dim = 768
            else:
                self.embedding_dim = 768  # Default for Google

        else:  # openai
            from langchain_openai import OpenAIEmbeddings

            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key
            )
            # OpenAI embedding dimensions
            if "large" in self.model_name:
                self.embedding_dim = 3072
            else:
                self.embedding_dim = 1536

        print(f"Da khoi tao model. Provider: {self.provider}, Dimension: {self.embedding_dim}")

    def _init_local_embeddings(self):
        """Initialize local embedding model (Sentence-BERT/E5)"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers chua duoc cai dat. "
                "Chay: pip install sentence-transformers"
            )

        # Determine device
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model tren {self.device}...")

        # Load model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embeddings = None  # Local model doesn't use LangChain embeddings

        # Get embedding dimension
        if self.model_name in LOCAL_MODELS:
            self.embedding_dim = LOCAL_MODELS[self.model_name]
        else:
            # Try to infer from model
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Check if E5 model (requires special prefix)
        self.is_e5 = "e5" in self.model_name.lower()
        if self.is_e5:
            print("E5 model detected - se them prefix 'query:' va 'passage:' tu dong")

    def _get_cache_key(self, text: str) -> str:
        """Tao cache key tu text"""
        # Include provider and model in cache key
        key_str = f"{self.provider}:{self.model_name}:{text}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self):
        """Load cache tu file"""
        cache_file = self.cache_dir / "embedding_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
                print(f"Da load {len(self._cache)} cached embeddings")
            except Exception as e:
                print(f"Loi khi load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Luu cache ra file"""
        if self.cache_dir:
            cache_file = self.cache_dir / "embedding_cache.json"
            with open(cache_file, "w") as f:
                json.dump(self._cache, f)

    def encode_text(
        self,
        text: Union[str, List[str]],
        batch_size: int = 100,
        show_progress: bool = True,
        use_cache: bool = True,
        is_query: bool = False  # For E5 models: True for query, False for passage
    ) -> np.ndarray:
        """
        Tao embeddings cho text hoac list of texts

        Args:
            text: Text hoac list of texts can encode
            batch_size: Batch size cho encoding
            show_progress: Hien thi progress bar
            use_cache: Su dung cache hay khong
            is_query: Danh dau la query (cho E5 models)

        Returns:
            Numpy array cua embeddings
        """
        # Convert single text to list
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = text
            single = False

        embeddings = []
        texts_to_embed = []
        text_indices = []

        # Check cache
        for i, t in enumerate(texts):
            cache_key = self._get_cache_key(t)
            if use_cache and cache_key in self._cache:
                embeddings.append((i, self._cache[cache_key]))
            else:
                texts_to_embed.append(t)
                text_indices.append(i)

        # Embed texts that are not in cache
        if texts_to_embed:
            new_embeddings = self._batch_embed_with_retry(
                texts_to_embed,
                batch_size,
                show_progress,
                is_query=is_query
            )

            # Add to result and cache
            for idx, emb in zip(text_indices, new_embeddings):
                embeddings.append((idx, emb))
                if use_cache:
                    cache_key = self._get_cache_key(texts[idx])
                    self._cache[cache_key] = emb

            # Save cache
            if use_cache and self.cache_dir:
                self._save_cache()

        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        result = np.array([e[1] for e in embeddings])

        return result[0] if single else result

    def _batch_embed_with_retry(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        max_retries: int = 3,
        is_query: bool = False
    ) -> List[List[float]]:
        """
        Batch embedding voi retry logic va backoff
        """
        all_embeddings = []

        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        iterator = tqdm(batches, desc="Embedding batches") if show_progress else batches

        for batch in iterator:
            for attempt in range(max_retries):
                try:
                    if self.provider == "local":
                        # Local embedding (Sentence-BERT/E5)
                        batch_embeddings = self._local_embed(batch, is_query=is_query)
                    else:
                        # Cloud embedding (OpenAI/Google)
                        batch_embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Error occurred, waiting {wait_time}s... (attempt {attempt + 1})")
                        time.sleep(wait_time)
                    else:
                        raise e

        return all_embeddings

    def _local_embed(
        self,
        texts: List[str],
        is_query: bool = False
    ) -> List[List[float]]:
        """
        Embed texts using local model (Sentence-BERT/E5)
        """
        # E5 models require prefix
        if self.is_e5:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]

        # Encode using sentence-transformers
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        return embeddings.tolist()

    def encode_chunks(
        self,
        chunks: List[Dict],
        text_field: str = "text",
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Tao embeddings cho list cac chunks

        Args:
            chunks: List cac chunks (dict voi text field)
            text_field: Ten field chua text trong chunk dict
            batch_size: Batch size cho encoding

        Returns:
            List chunks da duoc them embeddings
        """
        texts = [chunk[text_field] for chunk in chunks]

        print(f"Dang tao embeddings cho {len(texts)} chunks...")

        embeddings = self.encode_text(
            texts,
            batch_size=batch_size,
            show_progress=True
        )

        chunks_with_embeddings = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_data = chunk.copy()
            chunk_data["embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            chunk_data["embedding_model"] = self.model_name
            chunk_data["embedding_provider"] = self.provider
            chunks_with_embeddings.append(chunk_data)

        print(f"Da tao xong {len(chunks_with_embeddings)} embeddings")
        return chunks_with_embeddings

    def encode_query(self, query: str) -> List[float]:
        """
        Tao embedding cho query

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        if self.provider == "local":
            # Use local model with query prefix for E5
            embeddings = self._local_embed([query], is_query=True)
            return embeddings[0]
        else:
            return self.embeddings.embed_query(query)

    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, List[float]],
        embedding2: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Tinh cosine similarity giua 2 embeddings
        """
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)

        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        return float(similarity)

    def find_most_similar(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        candidate_embeddings: List[Union[np.ndarray, List[float]]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Tim top-k embeddings tuong dong nhat voi query
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        similarities = []
        for idx, candidate in enumerate(candidate_embeddings):
            if isinstance(candidate, list):
                candidate = np.array(candidate)

            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append({"index": idx, "similarity": sim})

        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def save_embeddings(
        self,
        chunks_with_embeddings: List[Dict],
        output_path: Union[str, Path]
    ) -> Path:
        """Luu chunks voi embeddings ra file JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)

        print(f"Da luu {len(chunks_with_embeddings)} embeddings tai: {output_path}")
        return output_path

    def load_embeddings(
        self,
        input_path: Union[str, Path]
    ) -> List[Dict]:
        """Load chunks voi embeddings tu file JSON"""
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Da load {len(chunks)} embeddings tu: {input_path}")
        return chunks


# Utility functions
def list_available_models():
    """List all available local models"""
    print("\n=== AVAILABLE LOCAL MODELS ===")
    print("\nSentence-BERT models:")
    for name, dim in LOCAL_MODELS.items():
        if "sentence-transformers" in name:
            print(f"  - {name} ({dim}d)")

    print("\nE5 models:")
    for name, dim in LOCAL_MODELS.items():
        if "e5" in name.lower():
            print(f"  - {name} ({dim}d)")

    print("\nVietnamese-specific models:")
    for name, dim in LOCAL_MODELS.items():
        if "vietnamese" in name.lower() or "vi" in name.lower():
            print(f"  - {name} ({dim}d)")

    print("\nAliases (shortcuts):")
    for alias, full_name in MODEL_ALIASES.items():
        print(f"  - '{alias}' -> {full_name}")


# Test function
if __name__ == "__main__":
    import os

    sample_texts = [
        "Day la cau thu nhat ve AI.",
        "Tri tue nhan tao dang phat trien nhanh.",
        "Hom nay troi dep qua."
    ]

    # Test local embedding first (no API key needed)
    print("=" * 60)
    print("Testing LOCAL Embeddings (Sentence-BERT)")
    print("=" * 60)

    try:
        embedder = TextEmbedding(provider="local", model_name="sbert")

        embeddings = embedder.encode_text(sample_texts, show_progress=False)
        print(f"Shape: {embeddings.shape}")

        sim = embedder.compute_similarity(embeddings[0], embeddings[1])
        print(f"Similarity (AI text 1, AI text 2): {sim:.4f}")

        sim = embedder.compute_similarity(embeddings[0], embeddings[2])
        print(f"Similarity (AI text 1, weather text): {sim:.4f}")

        # Test query encoding
        query_emb = embedder.encode_query("AI la gi?")
        print(f"Query embedding length: {len(query_emb)}")

        print("\n[PASS] Local Embedding (Sentence-BERT) works!")

    except Exception as e:
        print(f"[FAIL] Local embedding error: {e}")
        import traceback
        traceback.print_exc()

    # Test E5 model
    print("\n" + "=" * 60)
    print("Testing LOCAL Embeddings (E5)")
    print("=" * 60)

    try:
        embedder_e5 = TextEmbedding(provider="local", model_name="e5")

        embeddings_e5 = embedder_e5.encode_text(sample_texts, show_progress=False)
        print(f"Shape: {embeddings_e5.shape}")

        sim = embedder_e5.compute_similarity(embeddings_e5[0], embeddings_e5[1])
        print(f"Similarity (AI text 1, AI text 2): {sim:.4f}")

        print("\n[PASS] Local Embedding (E5) works!")

    except Exception as e:
        print(f"[FAIL] E5 embedding error: {e}")
        import traceback
        traceback.print_exc()

    # Test cloud embeddings if API keys available
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if google_key:
        print("\n" + "=" * 60)
        print("Testing GOOGLE Embeddings")
        print("=" * 60)
        try:
            embedder = TextEmbedding(provider="google")
            embeddings = embedder.encode_text(sample_texts[:2], show_progress=False)
            print(f"Shape: {embeddings.shape}")
            print("[PASS] Google Embedding works!")
        except Exception as e:
            print(f"[FAIL] Google embedding error: {e}")

    if openai_key:
        print("\n" + "=" * 60)
        print("Testing OPENAI Embeddings")
        print("=" * 60)
        try:
            embedder = TextEmbedding(provider="openai")
            embeddings = embedder.encode_text(sample_texts[:2], show_progress=False)
            print(f"Shape: {embeddings.shape}")
            print("[PASS] OpenAI Embedding works!")
        except Exception as e:
            print(f"[FAIL] OpenAI embedding error: {e}")

    # List available models
    list_available_models()
