"""
Embedding Module - Tao vector embeddings cho van ban
Ho tro ca OpenAI va Google Embeddings thong qua LangChain
"""

import os
import hashlib
import json
from typing import List, Dict, Union, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time


class TextEmbedding:
    """
    Lop tao embeddings cho van ban - ho tro ca OpenAI va Google
    - OpenAI: text-embedding-3-small (1536d), text-embedding-3-large (3072d)
    - Google: models/text-embedding-004 (768d), models/embedding-001 (768d)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: str = "google",  # "openai" or "google"
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Khoi tao Text Embedding model

        Args:
            model_name: Ten model (neu None se dung default theo provider)
            provider: "openai" hoac "google"
            api_key: API key (neu khong truyen se lay tu env)
            cache_dir: Thu muc cache embeddings
        """
        self.provider = provider.lower()

        # Default models
        if model_name is None:
            if self.provider == "google":
                model_name = "models/text-embedding-004"
            else:
                model_name = "text-embedding-3-small"

        self.model_name = model_name

        # Get API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            key_name = "GOOGLE_API_KEY" if self.provider == "google" else "OPENAI_API_KEY"
            raise ValueError(f"{key_name} chua duoc cau hinh!")

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

        if self.provider == "google":
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
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Tao embeddings cho text hoac list of texts

        Args:
            text: Text hoac list of texts can encode
            batch_size: Batch size cho encoding
            show_progress: Hien thi progress bar
            use_cache: Su dung cache hay khong

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
                show_progress
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
        max_retries: int = 3
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
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1})")
                        time.sleep(wait_time)
                    else:
                        raise e

        return all_embeddings

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


# Test function
if __name__ == "__main__":
    import os

    # Check which API key is available
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if google_key:
        print("Testing with Google Embeddings...")
        embedder = TextEmbedding(provider="google")

        sample_texts = [
            "Day la cau thu nhat ve AI.",
            "Tri tue nhan tao dang phat trien nhanh.",
            "Hom nay troi dep qua."
        ]

        embeddings = embedder.encode_text(sample_texts)
        print(f"Shape: {embeddings.shape}")

        sim = embedder.compute_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between text 1 and 2: {sim:.4f}")

        sim = embedder.compute_similarity(embeddings[0], embeddings[2])
        print(f"Similarity between text 1 and 3: {sim:.4f}")

        print("\nEmbedding Module (Google) initialized successfully!")

    elif openai_key:
        print("Testing with OpenAI Embeddings...")
        embedder = TextEmbedding(provider="openai")

        sample_texts = [
            "Day la cau thu nhat ve AI.",
            "Tri tue nhan tao dang phat trien nhanh."
        ]

        embeddings = embedder.encode_text(sample_texts)
        print(f"Shape: {embeddings.shape}")

        print("\nEmbedding Module (OpenAI) initialized successfully!")

    else:
        print("Khong tim thay GOOGLE_API_KEY hoac OPENAI_API_KEY!")
        print("Hay them vao file .env:")
        print("  GOOGLE_API_KEY=your_key")
        print("  hoac")
        print("  OPENAI_API_KEY=your_key")
