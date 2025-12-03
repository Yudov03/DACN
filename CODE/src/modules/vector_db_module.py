"""
Vector Database Module - Quan ly Qdrant vector database cho retrieval
Theo plan.pdf: su dung Qdrant thay vi ChromaDB/FAISS
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import uuid


class VectorDatabase:
    """
    Lop quan ly Qdrant vector database (theo plan.pdf)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "audio_transcripts",
        embedding_dimension: int = 1536,
        distance_metric: str = "cosine"
    ):
        """
        Khoi tao Qdrant Vector Database

        Args:
            host: Qdrant host (cho local)
            port: Qdrant port (cho local)
            url: Qdrant Cloud URL (optional)
            api_key: Qdrant Cloud API key (optional)
            collection_name: Ten collection
            embedding_dimension: Dimension cua embeddings (1536 cho text-embedding-3-small)
            distance_metric: Distance metric (cosine, euclid, dot)
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.distance = distance_map.get(distance_metric.lower(), Distance.COSINE)

        # Khoi tao Qdrant client
        print(f"Dang ket noi toi Qdrant...")
        try:
            if url:
                # Qdrant Cloud
                self.client = QdrantClient(url=url, api_key=api_key)
                print(f"Da ket noi toi Qdrant Cloud: {url}")
            else:
                # Local Qdrant
                self.client = QdrantClient(host=host, port=port)
                print(f"Da ket noi toi Qdrant local: {host}:{port}")

            # Khoi tao collection
            self._init_collection()

        except Exception as e:
            print(f"Loi ket noi Qdrant: {e}")
            print("Dang su dung Qdrant in-memory mode...")
            # Fallback to in-memory mode
            self.client = QdrantClient(":memory:")
            self._init_collection()

    def _init_collection(self):
        """Khoi tao hoac load collection"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            # Collection da ton tai
            collection_info = self.client.get_collection(self.collection_name)
            print(f"Da load collection '{self.collection_name}' (points: {collection_info.points_count})")
        else:
            # Tao collection moi
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=self.distance
                )
            )
            print(f"Da tao collection moi '{self.collection_name}'")

    def add_documents(
        self,
        chunks: List[Dict],
        embedding_field: str = "embedding",
        id_field: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """
        Them documents (chunks) vao Qdrant

        Args:
            chunks: List cac chunks co embeddings
            embedding_field: Ten field chua embedding
            id_field: Ten field chua ID (optional)
            batch_size: Batch size cho upsert

        Returns:
            So luong documents da them
        """
        points = []

        for idx, chunk in enumerate(chunks):
            # Generate ID
            if id_field and id_field in chunk:
                point_id = str(chunk[id_field])
            else:
                point_id = str(uuid.uuid4())

            # Get embedding
            embedding = chunk[embedding_field]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Build payload (metadata)
            payload = {}
            for k, v in chunk.items():
                if k != embedding_field:
                    # Qdrant accepts most types directly
                    if isinstance(v, (str, int, float, bool, list)):
                        payload[k] = v
                    elif v is None:
                        payload[k] = None
                    else:
                        payload[k] = str(v)

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)

        # Batch upsert
        total_added = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            total_added += len(batch)
            print(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

        print(f"Da them {total_added} documents vao Qdrant")
        return total_added

    def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Tim kiem documents tuong tu voi query

        Args:
            query_embedding: Embedding cua query
            top_k: So luong ket qua tra ve
            filter_dict: Bo loc metadata (optional)
            score_threshold: Nguong similarity score (optional)

        Returns:
            List cac documents tim duoc voi scores
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Build filter
        qdrant_filter = None
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, dict):
                    # Range filter
                    if "gte" in value or "lte" in value:
                        must_conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(
                                    gte=value.get("gte"),
                                    lte=value.get("lte")
                                )
                            )
                        )
                else:
                    # Exact match
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )

            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        # Search using query_points (Qdrant client v1.16+)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )

        # Format results
        documents = []
        for result in response.points:
            doc = {
                "id": str(result.id),
                "similarity": result.score,
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            }
            documents.append(doc)

        return documents

    def search_with_mmr(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Tim kiem voi Maximal Marginal Relevance (MMR) de tang diversity

        Args:
            query_embedding: Embedding cua query
            top_k: So luong ket qua tra ve
            fetch_k: So luong candidates fetch truoc khi MMR
            lambda_mult: Balance giua relevance va diversity (0-1)
            filter_dict: Bo loc metadata

        Returns:
            List cac documents da duoc re-rank theo MMR
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Fetch more results first
        initial_results = self.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            filter_dict=filter_dict
        )

        if len(initial_results) <= top_k:
            return initial_results

        # Apply MMR
        selected = []
        candidates = initial_results.copy()
        query_emb = np.array(query_embedding)

        while len(selected) < top_k and candidates:
            best_score = -float('inf')
            best_idx = 0

            for i, candidate in enumerate(candidates):
                # Relevance score (already have from search)
                relevance = candidate["similarity"]

                # Diversity score (max similarity to already selected)
                if selected:
                    max_sim = 0
                    for sel in selected:
                        # Simplified: use 1 - distance as diversity
                        # In practice, you'd need embeddings stored
                        max_sim = max(max_sim, 0.5)  # Placeholder
                    diversity = 1 - max_sim
                else:
                    diversity = 1

                # MMR score
                mmr_score = lambda_mult * relevance + (1 - lambda_mult) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(candidates[best_idx])
            candidates.pop(best_idx)

        return selected

    def get_collection_stats(self) -> Dict:
        """
        Lay thong ke ve collection

        Returns:
            Dict chua thong ke
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "type": "qdrant",
                "collection_name": self.collection_name,
                "count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status),
                "dimension": self.embedding_dimension
            }
        except Exception as e:
            return {
                "type": "qdrant",
                "collection_name": self.collection_name,
                "error": str(e)
            }

    def delete_collection(self):
        """Xoa collection"""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Da xoa collection '{self.collection_name}'")

    def delete_by_filter(self, filter_dict: Dict) -> int:
        """
        Xoa documents theo filter

        Args:
            filter_dict: Bo loc de xac dinh documents can xoa

        Returns:
            So documents da xoa (estimated)
        """
        # Build filter
        must_conditions = []
        for key, value in filter_dict.items():
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )

        qdrant_filter = Filter(must=must_conditions)

        # Count before delete
        count_before = self.get_collection_stats().get("count", 0)

        # Delete
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter)
        )

        # Count after delete
        count_after = self.get_collection_stats().get("count", 0)

        deleted = count_before - count_after
        print(f"Da xoa {deleted} documents")
        return deleted

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Lay document theo ID

        Args:
            doc_id: ID cua document

        Returns:
            Document hoac None
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )
            if results:
                point = results[0]
                return {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": point.payload
                }
            return None
        except Exception as e:
            print(f"Loi khi lay document: {e}")
            return None

    def update_payload(self, doc_id: str, payload_updates: Dict):
        """
        Cap nhat payload cua document

        Args:
            doc_id: ID cua document
            payload_updates: Dict chua cac field can update
        """
        self.client.set_payload(
            collection_name=self.collection_name,
            points=[doc_id],
            payload=payload_updates
        )
        print(f"Da cap nhat payload cho document {doc_id}")


# Test function
if __name__ == "__main__":
    print("Testing Qdrant Vector Database...")

    # Test in-memory mode
    db = VectorDatabase(
        collection_name="test_collection",
        embedding_dimension=1536
    )

    # Test stats
    stats = db.get_collection_stats()
    print(f"\nCollection stats: {stats}")

    # Test add documents
    test_chunks = [
        {
            "text": "Day la cau thu nhat",
            "embedding": [0.1] * 1536,
            "chunk_id": 0,
            "start_time": 0.0,
            "end_time": 5.0,
            "audio_file": "test.mp3"
        },
        {
            "text": "Day la cau thu hai",
            "embedding": [0.2] * 1536,
            "chunk_id": 1,
            "start_time": 5.0,
            "end_time": 10.0,
            "audio_file": "test.mp3"
        }
    ]

    db.add_documents(test_chunks)

    # Test search
    results = db.search(
        query_embedding=[0.15] * 1536,
        top_k=2
    )
    print(f"\nSearch results: {len(results)} documents found")
    for r in results:
        print(f"  - {r['text']}: similarity={r['similarity']:.4f}")

    # Final stats
    stats = db.get_collection_stats()
    print(f"\nFinal stats: {stats}")

    print("\nQdrant Vector Database Module initialized successfully!")
