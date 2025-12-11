"""
Caching Module - Cache embeddings va LLM responses de giam latency
Ho tro:
- In-memory cache (LRU)
- Disk cache (SQLite/JSON)
- Embedding cache (vector cache)
- Response cache (LLM output)
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import hashlib
import json
import time
import os
from pathlib import Path
from functools import lru_cache
import threading
from collections import OrderedDict


class LRUCache:
    """
    Thread-safe LRU Cache cho in-memory caching
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        Args:
            max_size: So luong items toi da
            ttl: Time-to-live (seconds), None = khong het han
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, key: Any) -> str:
        """Tao cache key tu bat ky input nao"""
        if isinstance(key, str):
            return key
        elif isinstance(key, (list, dict)):
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()

    def get(self, key: Any) -> Optional[Any]:
        """Lay value tu cache"""
        cache_key = self._make_key(key)

        with self.lock:
            if cache_key not in self.cache:
                self.misses += 1
                return None

            # Check TTL
            if self.ttl:
                if time.time() - self.timestamps[cache_key] > self.ttl:
                    del self.cache[cache_key]
                    del self.timestamps[cache_key]
                    self.misses += 1
                    return None

            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            self.hits += 1
            return self.cache[cache_key]

    def set(self, key: Any, value: Any) -> None:
        """Luu value vao cache"""
        cache_key = self._make_key(key)

        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.timestamps:
                    del self.timestamps[oldest_key]

            self.cache[cache_key] = value
            self.timestamps[cache_key] = time.time()

    def delete(self, key: Any) -> bool:
        """Xoa item khoi cache"""
        cache_key = self._make_key(key)

        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                if cache_key in self.timestamps:
                    del self.timestamps[cache_key]
                return True
            return False

    def clear(self) -> None:
        """Xoa toan bo cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict:
        """Lay thong ke cache"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "ttl": self.ttl
            }


class DiskCache:
    """
    Persistent disk cache su dung JSON files
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        max_size_mb: int = 500,
        ttl: Optional[int] = 86400  # 24 hours
    ):
        """
        Args:
            cache_dir: Thu muc luu cache
            max_size_mb: Kich thuoc toi da (MB)
            ttl: Time-to-live (seconds)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl = ttl
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load index tu disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"entries": {}, "total_size": 0}
        return {"entries": {}, "total_size": 0}

    def _save_index(self) -> None:
        """Luu index xuong disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            print(f"Warning: Could not save cache index: {e}")

    def _make_key(self, key: Any) -> str:
        """Tao cache key"""
        if isinstance(key, str):
            key_str = key
        elif isinstance(key, (list, dict)):
            key_str = json.dumps(key, sort_keys=True)
        else:
            key_str = str(key)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _get_file_path(self, cache_key: str) -> Path:
        """Lay duong dan file cache"""
        # Phan tan files vao subfolders de tranh qua nhieu files trong 1 folder
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.json"

    def get(self, key: Any) -> Optional[Any]:
        """Lay value tu cache"""
        cache_key = self._make_key(key)

        entry = self.index["entries"].get(cache_key)
        if not entry:
            return None

        # Check TTL
        if self.ttl and time.time() - entry["timestamp"] > self.ttl:
            self.delete(key)
            return None

        # Load from disk
        file_path = self._get_file_path(cache_key)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("value")
            except Exception:
                return None

        return None

    def set(self, key: Any, value: Any) -> None:
        """Luu value vao cache"""
        cache_key = self._make_key(key)

        # Serialize value
        try:
            data = {"value": value, "timestamp": time.time()}
            serialized = json.dumps(data, ensure_ascii=False)
            size = len(serialized.encode('utf-8'))
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not serialize cache value: {e}")
            return

        # Check if need to cleanup
        if self.index["total_size"] + size > self.max_size_bytes:
            self._cleanup()

        # Save to disk
        file_path = self._get_file_path(cache_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(serialized)

            # Update index
            if cache_key in self.index["entries"]:
                self.index["total_size"] -= self.index["entries"][cache_key]["size"]

            self.index["entries"][cache_key] = {
                "timestamp": time.time(),
                "size": size
            }
            self.index["total_size"] += size
            self._save_index()
        except Exception as e:
            print(f"Warning: Could not write cache file: {e}")

    def delete(self, key: Any) -> bool:
        """Xoa item khoi cache"""
        cache_key = self._make_key(key)

        if cache_key in self.index["entries"]:
            entry = self.index["entries"][cache_key]
            self.index["total_size"] -= entry["size"]
            del self.index["entries"][cache_key]

            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()

            self._save_index()
            return True
        return False

    def _cleanup(self) -> None:
        """Xoa cac entries cu nhat khi vuot qua max size"""
        # Sort by timestamp
        sorted_entries = sorted(
            self.index["entries"].items(),
            key=lambda x: x[1]["timestamp"]
        )

        # Remove oldest until under limit
        target_size = self.max_size_bytes * 0.8  # Free up to 80%
        for cache_key, entry in sorted_entries:
            if self.index["total_size"] <= target_size:
                break

            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()

            self.index["total_size"] -= entry["size"]
            del self.index["entries"][cache_key]

        self._save_index()

    def clear(self) -> None:
        """Xoa toan bo cache"""
        import shutil
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.name != "index.json":
                item.unlink()

        self.index = {"entries": {}, "total_size": 0}
        self._save_index()

    def get_stats(self) -> Dict:
        """Lay thong ke cache"""
        return {
            "entries": len(self.index["entries"]),
            "total_size_mb": self.index["total_size"] / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "ttl": self.ttl,
            "cache_dir": str(self.cache_dir)
        }


class EmbeddingCache:
    """
    Cache dac biet cho embeddings
    Ket hop in-memory va disk cache
    """

    def __init__(
        self,
        cache_dir: str = "./cache/embeddings",
        memory_cache_size: int = 5000,
        disk_cache_size_mb: int = 1000,
        ttl: Optional[int] = 604800  # 7 days
    ):
        """
        Args:
            cache_dir: Thu muc luu cache
            memory_cache_size: So luong embeddings trong memory
            disk_cache_size_mb: Kich thuoc disk cache (MB)
            ttl: Time-to-live (seconds)
        """
        self.memory_cache = LRUCache(max_size=memory_cache_size, ttl=ttl)
        self.disk_cache = DiskCache(
            cache_dir=cache_dir,
            max_size_mb=disk_cache_size_mb,
            ttl=ttl
        )

    def _make_key(self, text: str, model_name: str) -> str:
        """Tao cache key tu text va model"""
        key_str = f"{model_name}:{text}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Lay embedding tu cache

        Args:
            text: Text da embed
            model_name: Ten model embedding

        Returns:
            Embedding vector hoac None
        """
        cache_key = self._make_key(text, model_name)

        # Check memory first
        embedding = self.memory_cache.get(cache_key)
        if embedding is not None:
            return embedding

        # Check disk
        embedding = self.disk_cache.get(cache_key)
        if embedding is not None:
            # Promote to memory
            self.memory_cache.set(cache_key, embedding)
            return embedding

        return None

    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """
        Luu embedding vao cache

        Args:
            text: Text da embed
            model_name: Ten model embedding
            embedding: Embedding vector
        """
        cache_key = self._make_key(text, model_name)

        # Save to both caches
        self.memory_cache.set(cache_key, embedding)
        self.disk_cache.set(cache_key, embedding)

    def get_batch(
        self,
        texts: List[str],
        model_name: str
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Lay batch embeddings tu cache

        Args:
            texts: List texts
            model_name: Ten model

        Returns:
            Tuple (embeddings, missing_indices)
            embeddings: List embeddings (None cho cac missing)
            missing_indices: Indices cua cac texts khong co trong cache
        """
        embeddings = []
        missing_indices = []

        for i, text in enumerate(texts):
            emb = self.get(text, model_name)
            embeddings.append(emb)
            if emb is None:
                missing_indices.append(i)

        return embeddings, missing_indices

    def set_batch(
        self,
        texts: List[str],
        model_name: str,
        embeddings: List[List[float]]
    ) -> None:
        """Luu batch embeddings vao cache"""
        for text, emb in zip(texts, embeddings):
            self.set(text, model_name, emb)

    def get_stats(self) -> Dict:
        """Lay thong ke cache"""
        return {
            "memory": self.memory_cache.get_stats(),
            "disk": self.disk_cache.get_stats()
        }

    def clear(self) -> None:
        """Xoa toan bo cache"""
        self.memory_cache.clear()
        self.disk_cache.clear()


class ResponseCache:
    """
    Cache cho LLM responses
    Key = hash(prompt + params)
    """

    def __init__(
        self,
        cache_dir: str = "./cache/responses",
        memory_cache_size: int = 500,
        disk_cache_size_mb: int = 200,
        ttl: Optional[int] = 3600  # 1 hour (responses co the outdated)
    ):
        self.memory_cache = LRUCache(max_size=memory_cache_size, ttl=ttl)
        self.disk_cache = DiskCache(
            cache_dir=cache_dir,
            max_size_mb=disk_cache_size_mb,
            ttl=ttl
        )

    def _make_key(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Tao cache key tu prompt va params"""
        key_dict = {
            "prompt": prompt,
            "model": model_name,
            "temperature": temperature,
            **kwargs
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        **kwargs
    ) -> Optional[str]:
        """Lay response tu cache"""
        # Don't cache non-deterministic requests
        if temperature > 0.3:
            return None

        cache_key = self._make_key(prompt, model_name, temperature, **kwargs)

        # Check memory first
        response = self.memory_cache.get(cache_key)
        if response is not None:
            return response

        # Check disk
        response = self.disk_cache.get(cache_key)
        if response is not None:
            self.memory_cache.set(cache_key, response)
            return response

        return None

    def set(
        self,
        prompt: str,
        model_name: str,
        response: str,
        temperature: float = 0.7,
        **kwargs
    ) -> None:
        """Luu response vao cache"""
        # Don't cache non-deterministic requests
        if temperature > 0.3:
            return

        cache_key = self._make_key(prompt, model_name, temperature, **kwargs)
        self.memory_cache.set(cache_key, response)
        self.disk_cache.set(cache_key, response)

    def get_stats(self) -> Dict:
        """Lay thong ke cache"""
        return {
            "memory": self.memory_cache.get_stats(),
            "disk": self.disk_cache.get_stats()
        }

    def clear(self) -> None:
        """Xoa toan bo cache"""
        self.memory_cache.clear()
        self.disk_cache.clear()


class CacheManager:
    """
    Manager tong hop quan ly tat ca caches
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        enable_embedding_cache: bool = True,
        enable_response_cache: bool = True,
        memory_cache_size: int = 1000,
        disk_cache_size_mb: int = 500
    ):
        """
        Args:
            cache_dir: Thu muc goc cho cache
            enable_embedding_cache: Bat cache embeddings
            enable_response_cache: Bat cache LLM responses
            memory_cache_size: Kich thuoc memory cache
            disk_cache_size_mb: Kich thuoc disk cache (MB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_cache = None
        self.response_cache = None

        if enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(
                cache_dir=str(self.cache_dir / "embeddings"),
                memory_cache_size=memory_cache_size * 5,  # More for embeddings
                disk_cache_size_mb=disk_cache_size_mb * 2
            )

        if enable_response_cache:
            self.response_cache = ResponseCache(
                cache_dir=str(self.cache_dir / "responses"),
                memory_cache_size=memory_cache_size,
                disk_cache_size_mb=disk_cache_size_mb
            )

        # General purpose cache
        self.general_cache = LRUCache(max_size=memory_cache_size)

    def get_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Lay embedding tu cache"""
        if self.embedding_cache:
            return self.embedding_cache.get(text, model_name)
        return None

    def set_embedding(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Luu embedding vao cache"""
        if self.embedding_cache:
            self.embedding_cache.set(text, model_name, embedding)

    def get_response(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Lay LLM response tu cache"""
        if self.response_cache:
            return self.response_cache.get(prompt, model_name, temperature)
        return None

    def set_response(
        self,
        prompt: str,
        model_name: str,
        response: str,
        temperature: float = 0.7
    ) -> None:
        """Luu LLM response vao cache"""
        if self.response_cache:
            self.response_cache.set(prompt, model_name, response, temperature)

    def cached_function(self, ttl: Optional[int] = None):
        """
        Decorator de cache function results

        Usage:
            @cache_manager.cached_function(ttl=3600)
            def my_function(arg1, arg2):
                ...
        """
        def decorator(func: Callable) -> Callable:
            func_cache = LRUCache(max_size=1000, ttl=ttl)

            def wrapper(*args, **kwargs):
                key = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }
                result = func_cache.get(key)
                if result is not None:
                    return result

                result = func(*args, **kwargs)
                func_cache.set(key, result)
                return result

            return wrapper
        return decorator

    def get_stats(self) -> Dict:
        """Lay thong ke tat ca caches"""
        stats = {
            "general": self.general_cache.get_stats()
        }
        if self.embedding_cache:
            stats["embedding"] = self.embedding_cache.get_stats()
        if self.response_cache:
            stats["response"] = self.response_cache.get_stats()
        return stats

    def clear_all(self) -> None:
        """Xoa tat ca caches"""
        self.general_cache.clear()
        if self.embedding_cache:
            self.embedding_cache.clear()
        if self.response_cache:
            self.response_cache.clear()

    def print_stats(self) -> None:
        """In thong ke caches"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("CACHE STATISTICS")
        print("=" * 50)

        for cache_name, cache_stats in stats.items():
            print(f"\n{cache_name.upper()} Cache:")
            if isinstance(cache_stats, dict):
                if "memory" in cache_stats:
                    # Nested cache (embedding/response)
                    mem = cache_stats["memory"]
                    disk = cache_stats["disk"]
                    print(f"  Memory: {mem['size']}/{mem['max_size']} items, "
                          f"hit rate: {mem['hit_rate']:.2%}")
                    print(f"  Disk: {disk['entries']} entries, "
                          f"{disk['total_size_mb']:.2f}/{disk['max_size_mb']:.0f} MB")
                else:
                    # Simple cache
                    print(f"  Size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")
                    print(f"  Hit rate: {cache_stats.get('hit_rate', 0):.2%}")


# Test
if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Testing Caching Module")
    print("=" * 60)

    # Test LRU Cache
    print("\n--- LRU Cache ---")
    cache = LRUCache(max_size=3, ttl=10)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    print(f"Get key1: {cache.get('key1')}")
    print(f"Get key2: {cache.get('key2')}")
    print(f"Stats: {cache.get_stats()}")

    # Add 4th item (should evict oldest)
    cache.set("key4", "value4")
    print(f"After adding key4, get key3: {cache.get('key3')}")  # Should be None

    # Test Embedding Cache
    print("\n--- Embedding Cache ---")
    emb_cache = EmbeddingCache(
        cache_dir="./test_cache/embeddings",
        memory_cache_size=100
    )

    # Simulate embedding
    test_embedding = np.random.randn(768).tolist()
    emb_cache.set("Hello world", "test-model", test_embedding)

    cached_emb = emb_cache.get("Hello world", "test-model")
    print(f"Cached embedding length: {len(cached_emb) if cached_emb else 0}")
    print(f"Cache stats: {emb_cache.get_stats()}")

    # Test Cache Manager
    print("\n--- Cache Manager ---")
    manager = CacheManager(cache_dir="./test_cache")

    # Cache embedding
    manager.set_embedding("Test text", "model1", test_embedding)
    cached = manager.get_embedding("Test text", "model1")
    print(f"Manager cached embedding: {len(cached) if cached else 0}")

    # Cache response
    manager.set_response("What is AI?", "gpt-3.5", "AI is...", temperature=0.0)
    cached_resp = manager.get_response("What is AI?", "gpt-3.5", temperature=0.0)
    print(f"Manager cached response: {cached_resp[:20] if cached_resp else None}...")

    manager.print_stats()

    # Cleanup
    import shutil
    shutil.rmtree("./test_cache", ignore_errors=True)

    print("\n" + "=" * 60)
    print("Caching Module ready!")
