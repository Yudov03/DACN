"""
Embedding Module - Tạo vector embeddings cho văn bản
Sử dụng Sentence-BERT hoặc các model embedding khác
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union, Optional
from pathlib import Path
import json
from tqdm import tqdm


class TextEmbedding:
    """
    Lớp tạo embeddings cho văn bản sử dụng Sentence-BERT
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None
    ):
        """
        Khởi tạo Text Embedding model

        Args:
            model_name: Tên model từ sentence-transformers
            device: Device để chạy model (cuda/cpu)
        """
        self.model_name = model_name

        print(f"Đang tải embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Đã tải xong model. Embedding dimension: {self.embedding_dim}")

    def encode_text(
        self,
        text: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Tạo embeddings cho text hoặc list of texts

        Args:
            text: Text hoặc list of texts cần encode
            batch_size: Batch size cho encoding
            show_progress: Hiển thị progress bar
            normalize: Normalize embeddings về unit length

        Returns:
            Numpy array của embeddings
        """
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def encode_chunks(
        self,
        chunks: List[Dict],
        text_field: str = "text",
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Tạo embeddings cho list các chunks

        Args:
            chunks: List các chunks (dict với text field)
            text_field: Tên field chứa text trong chunk dict
            batch_size: Batch size cho encoding

        Returns:
            List chunks đã được thêm embeddings
        """
        # Extract texts
        texts = [chunk[text_field] for chunk in chunks]

        print(f"Đang tạo embeddings cho {len(texts)} chunks...")

        # Generate embeddings
        embeddings = self.encode_text(
            texts,
            batch_size=batch_size,
            show_progress=True
        )

        # Add embeddings to chunks
        chunks_with_embeddings = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_data = chunk.copy()
            chunk_data["embedding"] = embedding.tolist()  # Convert to list for JSON serialization
            chunk_data["embedding_model"] = self.model_name
            chunks_with_embeddings.append(chunk_data)

        print(f"✓ Đã tạo xong {len(chunks_with_embeddings)} embeddings")
        return chunks_with_embeddings

    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, List[float]],
        embedding2: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Tính cosine similarity giữa 2 embeddings

        Args:
            embedding1: Embedding thứ nhất
            embedding2: Embedding thứ hai

        Returns:
            Cosine similarity score (0-1)
        """
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)

        # Cosine similarity
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
        Tìm top-k embeddings tương đồng nhất với query

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List các candidate embeddings
            top_k: Số lượng kết quả trả về

        Returns:
            List các dict chứa index và similarity score
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        similarities = []
        for idx, candidate in enumerate(candidate_embeddings):
            if isinstance(candidate, list):
                candidate = np.array(candidate)

            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append({"index": idx, "similarity": sim})

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def save_embeddings(
        self,
        chunks_with_embeddings: List[Dict],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Lưu chunks với embeddings ra file JSON

        Args:
            chunks_with_embeddings: List chunks có embeddings
            output_path: Đường dẫn file output

        Returns:
            Path của file đã lưu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)

        print(f"Đã lưu {len(chunks_with_embeddings)} embeddings tại: {output_path}")
        return output_path

    def load_embeddings(
        self,
        input_path: Union[str, Path]
    ) -> List[Dict]:
        """
        Load chunks với embeddings từ file JSON

        Args:
            input_path: Đường dẫn file input

        Returns:
            List chunks có embeddings
        """
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Đã load {len(chunks)} embeddings từ: {input_path}")
        return chunks

    def batch_encode_chunks_from_files(
        self,
        chunk_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        batch_size: int = 32
    ) -> List[Path]:
        """
        Encode embeddings cho nhiều file chunks

        Args:
            chunk_files: List các file chứa chunks
            output_dir: Thư mục lưu output
            batch_size: Batch size

        Returns:
            List paths của các file embeddings đã lưu
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for chunk_file in tqdm(chunk_files, desc="Processing chunk files"):
            chunk_file = Path(chunk_file)

            # Load chunks
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            # Encode
            chunks_with_embeddings = self.encode_chunks(chunks, batch_size=batch_size)

            # Save
            output_filename = chunk_file.stem + "_embeddings.json"
            output_path = output_dir / output_filename
            self.save_embeddings(chunks_with_embeddings, output_path)

            output_paths.append(output_path)

        print(f"\n✓ Hoàn thành! Đã tạo embeddings cho {len(output_paths)} files")
        return output_paths


# Test function
if __name__ == "__main__":
    # Example usage
    embedder = TextEmbedding()

    # Test với text đơn giản
    sample_texts = [
        "Đây là câu thứ nhất về AI.",
        "Trí tuệ nhân tạo đang phát triển nhanh.",
        "Hôm nay trời đẹp quá."
    ]

    embeddings = embedder.encode_text(sample_texts)
    print(f"Shape: {embeddings.shape}")

    # Test similarity
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 1 and 2: {sim:.4f}")

    sim = embedder.compute_similarity(embeddings[0], embeddings[2])
    print(f"Similarity between text 1 and 3: {sim:.4f}")

    print("\nEmbedding Module initialized successfully!")
