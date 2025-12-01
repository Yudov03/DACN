"""
Vector Database Module - Quản lý vector database cho retrieval
Hỗ trợ ChromaDB và FAISS
"""

import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime


class VectorDatabase:
    """
    Lớp quản lý vector database với ChromaDB hoặc FAISS
    """

    def __init__(
        self,
        db_type: str = "chromadb",
        db_path: Optional[Union[str, Path]] = None,
        collection_name: str = "audio_transcripts",
        embedding_dimension: int = 768
    ):
        """
        Khởi tạo Vector Database

        Args:
            db_type: Loại database (chromadb hoặc faiss)
            db_path: Đường dẫn lưu database
            collection_name: Tên collection
            embedding_dimension: Dimension của embeddings
        """
        self.db_type = db_type.lower()
        self.db_path = Path(db_path) if db_path else Path("./data/vector_db")
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        self.db_path.mkdir(parents=True, exist_ok=True)

        if self.db_type == "chromadb":
            self._init_chromadb()
        elif self.db_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"DB type '{db_type}' không hợp lệ. Chọn: chromadb hoặc faiss")

    def _init_chromadb(self):
        """Khởi tạo ChromaDB"""
        print(f"Khởi tạo ChromaDB tại {self.db_path}...")

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Đã load collection '{self.collection_name}' (items: {self.collection.count()})")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Audio transcript embeddings"}
            )
            print(f"Đã tạo collection mới '{self.collection_name}'")

    def _init_faiss(self):
        """Khởi tạo FAISS"""
        print(f"Khởi tạo FAISS index tại {self.db_path}...")

        self.index_path = self.db_path / f"{self.collection_name}.faiss"
        self.metadata_path = self.db_path / f"{self.collection_name}_metadata.pkl"

        # Load existing index hoặc tạo mới
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            print(f"Đã load FAISS index (vectors: {self.index.ntotal})")
        else:
            # Tạo index mới - sử dụng IndexFlatIP cho cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            print(f"Đã tạo FAISS index mới (dimension: {self.embedding_dimension})")

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, "rb") as f:
                self.metadata_store = pickle.load(f)
        else:
            self.metadata_store = []

    def add_documents(
        self,
        chunks: List[Dict],
        embedding_field: str = "embedding",
        id_field: Optional[str] = None
    ) -> int:
        """
        Thêm documents (chunks) vào database

        Args:
            chunks: List các chunks có embeddings
            embedding_field: Tên field chứa embedding
            id_field: Tên field chứa ID (optional)

        Returns:
            Số lượng documents đã thêm
        """
        if self.db_type == "chromadb":
            return self._add_to_chromadb(chunks, embedding_field, id_field)
        else:  # faiss
            return self._add_to_faiss(chunks, embedding_field)

    def _add_to_chromadb(
        self,
        chunks: List[Dict],
        embedding_field: str,
        id_field: Optional[str]
    ) -> int:
        """Thêm documents vào ChromaDB"""
        # Prepare data
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            # Generate ID
            if id_field and id_field in chunk:
                doc_id = str(chunk[id_field])
            else:
                doc_id = f"doc_{datetime.now().timestamp()}_{idx}"

            ids.append(doc_id)
            embeddings.append(chunk[embedding_field])
            documents.append(chunk.get("text", ""))

            # Metadata (exclude embedding và text)
            metadata = {k: v for k, v in chunk.items()
                       if k not in [embedding_field, "text"] and isinstance(v, (str, int, float, bool))}
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"✓ Đã thêm {len(ids)} documents vào ChromaDB")
        return len(ids)

    def _add_to_faiss(self, chunks: List[Dict], embedding_field: str) -> int:
        """Thêm documents vào FAISS"""
        # Extract embeddings
        embeddings = np.array([chunk[embedding_field] for chunk in chunks], dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store metadata
        for chunk in chunks:
            # Remove embedding from metadata to save space
            metadata = {k: v for k, v in chunk.items() if k != embedding_field}
            self.metadata_store.append(metadata)

        # Save index and metadata
        self._save_faiss()

        print(f"✓ Đã thêm {len(chunks)} documents vào FAISS")
        return len(chunks)

    def _save_faiss(self):
        """Lưu FAISS index và metadata"""
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata_store, f)

    def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Tìm kiếm documents tương tự với query

        Args:
            query_embedding: Embedding của query
            top_k: Số lượng kết quả trả về
            filter_dict: Bộ lọc metadata (chỉ cho ChromaDB)

        Returns:
            List các documents tìm được với scores
        """
        if self.db_type == "chromadb":
            return self._search_chromadb(query_embedding, top_k, filter_dict)
        else:  # faiss
            return self._search_faiss(query_embedding, top_k)

    def _search_chromadb(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int,
        filter_dict: Optional[Dict]
    ) -> List[Dict]:
        """Tìm kiếm trong ChromaDB"""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )

        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i],
                "similarity": 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            documents.append(doc)

        return documents

    def _search_faiss(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int
    ) -> List[Dict]:
        """Tìm kiếm trong FAISS"""
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Reshape to 2D array
        query_embedding = query_embedding.reshape(1, -1)

        # Normalize
        faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Format results
        documents = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata_store):
                doc = self.metadata_store[idx].copy()
                doc["similarity"] = float(distances[0][i])
                doc["index"] = int(idx)
                documents.append(doc)

        return documents

    def get_collection_stats(self) -> Dict:
        """
        Lấy thống kê về collection

        Returns:
            Dict chứa thống kê
        """
        if self.db_type == "chromadb":
            return {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "count": self.collection.count(),
                "path": str(self.db_path)
            }
        else:  # faiss
            return {
                "type": "faiss",
                "collection_name": self.collection_name,
                "count": self.index.ntotal,
                "dimension": self.embedding_dimension,
                "path": str(self.db_path)
            }

    def delete_collection(self):
        """Xóa collection"""
        if self.db_type == "chromadb":
            self.client.delete_collection(name=self.collection_name)
            print(f"Đã xóa collection '{self.collection_name}'")
        else:  # faiss
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            print(f"Đã xóa FAISS index '{self.collection_name}'")


# Test function
if __name__ == "__main__":
    # Example usage
    print("Testing ChromaDB:")
    db_chroma = VectorDatabase(db_type="chromadb", collection_name="test_collection")
    stats = db_chroma.get_collection_stats()
    print(stats)

    print("\nTesting FAISS:")
    db_faiss = VectorDatabase(db_type="faiss", collection_name="test_collection", embedding_dimension=768)
    stats = db_faiss.get_collection_stats()
    print(stats)

    print("\nVector Database Module initialized successfully!")
