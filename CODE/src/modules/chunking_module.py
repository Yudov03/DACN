"""
Chunking Module - Phan doan van ban theo ngu nghia
Theo plan.pdf: su dung LangChain SemanticChunker va RecursiveCharacterTextSplitter
"""

import re
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_experimental.text_splitter import SemanticChunker
    HAS_SEMANTIC_CHUNKER = True
except ImportError:
    HAS_SEMANTIC_CHUNKER = False
    SemanticChunker = None
import os


class TextChunker:
    """
    Lop xu ly chunking van ban voi nhieu phuong phap khac nhau
    Theo plan.pdf: ho tro RecursiveCharacterTextSplitter va SemanticChunker
    """

    def __init__(
        self,
        chunk_size: int = None,  # Reads from CHUNK_SIZE
        chunk_overlap: int = None,  # Reads from CHUNK_OVERLAP
        method: str = None,  # Reads from CHUNKING_METHOD
        semantic_threshold: float = None,  # Reads from SEMANTIC_THRESHOLD
        semantic_window_size: int = None,  # Reads from SEMANTIC_WINDOW_SIZE
        api_key: Optional[str] = None,
        embedding_provider: str = "local"  # "local", "google", or "openai"
    ):
        """
        Khoi tao Text Chunker

        Args:
            chunk_size: Kich thuoc chunk (so ky tu)
            chunk_overlap: So ky tu overlap giua cac chunks
            method: Phuong phap chunking (fixed, semantic, sentence, recursive)
            semantic_threshold: Nguong cosine similarity cho semantic chunking
            semantic_window_size: Window size cho semantic chunking
            api_key: API key (cho semantic chunking voi cloud providers)
            embedding_provider: "local" (SBERT/E5), "google", hoac "openai"
        """
        # Read from .env with defaults
        self.chunk_size = chunk_size if chunk_size is not None else int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP", "50"))
        self.method = method or os.getenv("CHUNKING_METHOD", "semantic")
        self.semantic_threshold = semantic_threshold if semantic_threshold is not None else float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))
        self.semantic_window_size = semantic_window_size if semantic_window_size is not None else int(os.getenv("SEMANTIC_WINDOW_SIZE", "5"))
        self.embedding_provider = embedding_provider.lower()

        # Get API key based on provider
        if api_key:
            self.api_key = api_key
        elif self.embedding_provider == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        valid_methods = ["fixed", "semantic", "sentence", "recursive"]
        if self.method not in valid_methods:
            raise ValueError(f"Method '{method}' khong hop le. Chon: {valid_methods}")

        # Khoi tao splitters
        self._init_splitters()

    def _init_splitters(self):
        """Khoi tao cac text splitters"""
        # RecursiveCharacterTextSplitter (LangChain) - dung cho fixed va recursive
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
            length_function=len
        )

        # SemanticChunker (LangChain) - chi khoi tao khi can
        self._semantic_chunker = None

    def _get_semantic_chunker(self):
        """Lazy load SemanticChunker - ho tro Local, OpenAI va Google"""
        if self._semantic_chunker is None:
            if self.embedding_provider == "local":
                # Use local embeddings - return None, will use _local_semantic_chunking
                return None

            if not self.api_key:
                key_name = "GOOGLE_API_KEY" if self.embedding_provider == "google" else "OPENAI_API_KEY"
                raise ValueError(f"{key_name} can thiet cho semantic chunking!")

            # Khoi tao embeddings dua tren provider
            if self.embedding_provider == "google":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=self.api_key
                )
            else:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=self.api_key
                )

            self._semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
        return self._semantic_chunker

    def _get_local_embedder(self):
        """Lazy load local TextEmbedding for semantic chunking"""
        if not hasattr(self, '_local_embedder') or self._local_embedder is None:
            from src.modules import TextEmbedding
            self._local_embedder = TextEmbedding(provider="local")
        return self._local_embedder

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chia van ban thanh cac chunks

        Args:
            text: Van ban can chunk
            metadata: Metadata bo sung cho chunks

        Returns:
            List cac chunks voi metadata
        """
        if self.method == "fixed":
            chunks = self._fixed_size_chunking(text)
        elif self.method == "sentence":
            chunks = self._sentence_based_chunking(text)
        elif self.method == "recursive":
            chunks = self._recursive_chunking(text)
        else:  # semantic
            chunks = self._semantic_chunking(text)

        # Them metadata
        return self._add_metadata(chunks, metadata or {})

    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Chunking voi kich thuoc co dinh

        Args:
            text: Van ban can chunk

        Returns:
            List cac chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Tranh cat giua tu
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    end = start + last_space

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c.strip()]

    def _sentence_based_chunking(self, text: str) -> List[str]:
        """
        Chunking dua tren cau (sentence boundaries)

        Args:
            text: Van ban can chunk

        Returns:
            List cac chunks
        """
        # Tach thanh cac cau
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Neu them cau nay vao chunk hien tai vuot qua chunk_size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Giu overlap
                current_chunk = current_chunk[-self.chunk_overlap:] + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Them chunk cuoi cung
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c.strip()]

    def _recursive_chunking(self, text: str) -> List[str]:
        """
        Chunking su dung LangChain RecursiveCharacterTextSplitter

        Args:
            text: Van ban can chunk

        Returns:
            List cac chunks
        """
        docs = self.recursive_splitter.create_documents([text])
        return [doc.page_content for doc in docs if doc.page_content.strip()]

    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Chunking dua tren ngu nghia su dung local embeddings hoac LangChain SemanticChunker

        Args:
            text: Van ban can chunk

        Returns:
            List cac chunks
        """
        try:
            # Use local semantic chunking for local provider
            if self.embedding_provider == "local":
                return self._local_semantic_chunking(text)

            # Use LangChain SemanticChunker for cloud providers
            semantic_chunker = self._get_semantic_chunker()
            docs = semantic_chunker.create_documents([text])
            chunks = [doc.page_content for doc in docs if doc.page_content.strip()]

            # Neu chunks qua ngan, gop lai
            merged_chunks = self._merge_short_chunks(chunks, min_length=100)
            return merged_chunks

        except Exception as e:
            print(f"Loi semantic chunking: {e}")
            print("Fallback sang recursive chunking...")
            return self._recursive_chunking(text)

    def _local_semantic_chunking(self, text: str) -> List[str]:
        """
        Semantic chunking su dung local embeddings (SBERT/E5)

        Algorithm:
        1. Split text into sentences
        2. Compute embeddings for each sentence
        3. Calculate cosine similarity between consecutive sentences
        4. Split when similarity drops below threshold

        Args:
            text: Van ban can chunk

        Returns:
            List cac chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Get embeddings for all sentences
        embedder = self._get_local_embedder()
        embeddings = embedder.encode_text(sentences, show_progress=False)

        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints where similarity drops significantly
        breakpoints = [0]  # Start of first chunk

        # Use percentile-based threshold (like LangChain)
        if similarities:
            threshold = np.percentile(similarities, 100 - 95)  # Bottom 5% similarities
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    breakpoints.append(i + 1)

        breakpoints.append(len(sentences))  # End of last chunk

        # Create chunks from breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1]
            chunk_text = " ".join(sentences[start:end])
            if chunk_text.strip():
                chunks.append(chunk_text)

        # Merge short chunks and split long chunks
        merged_chunks = self._merge_short_chunks(chunks, min_length=100)
        final_chunks = self._split_long_chunks(merged_chunks, max_length=self.chunk_size)

        return final_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Pattern for sentence endings (Vietnamese and English)
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _split_long_chunks(self, chunks: List[str], max_length: int) -> List[str]:
        """Split chunks that exceed max_length"""
        result = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                result.append(chunk)
            else:
                # Use recursive splitter for long chunks
                sub_chunks = self._recursive_chunking(chunk)
                result.extend(sub_chunks)
        return result

    def _merge_short_chunks(self, chunks: List[str], min_length: int = 100) -> List[str]:
        """
        Gop cac chunks qua ngan lai voi nhau

        Args:
            chunks: List cac chunks
            min_length: Do dai toi thieu cua chunk

        Returns:
            List chunks da gop
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            if len(current) < min_length:
                current += " " + chunk
            else:
                if current.strip():
                    merged.append(current.strip())
                current = chunk

        if current.strip():
            merged.append(current.strip())

        return merged

    def _add_metadata(self, chunks: List[str], base_metadata: Dict) -> List[Dict]:
        """
        Them metadata cho cac chunks

        Args:
            chunks: List cac chunks
            base_metadata: Metadata co ban

        Returns:
            List cac chunks voi metadata
        """
        chunks_with_metadata = []

        for idx, chunk_text in enumerate(chunks):
            chunk_data = {
                "chunk_id": idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                **base_metadata
            }
            chunks_with_metadata.append(chunk_data)

        return chunks_with_metadata

    def chunk_transcript(
        self,
        transcript_data: Dict,
        preserve_timestamps: bool = True
    ) -> List[Dict]:
        """
        Chunk transcript tu ASR output, giu lai timestamps

        Args:
            transcript_data: Du lieu transcript tu ASR
            preserve_timestamps: Co giu timestamps khong

        Returns:
            List cac chunks voi timestamps
        """
        if preserve_timestamps and "segments" in transcript_data:
            return self._chunk_with_timestamps(transcript_data)
        else:
            # Chunk toan bo full_text
            return self.chunk_text(
                transcript_data.get("full_text", ""),
                metadata={
                    "audio_file": transcript_data.get("audio_filename", ""),
                    "source": "transcript"
                }
            )

    def _chunk_with_timestamps(self, transcript_data: Dict) -> List[Dict]:
        """
        Chunk transcript giu nguyen timestamps tu cac segments

        Args:
            transcript_data: Du lieu transcript

        Returns:
            List cac chunks voi timestamps
        """
        segments = transcript_data.get("segments", [])
        if not segments:
            return self.chunk_text(
                transcript_data.get("full_text", ""),
                metadata={"audio_file": transcript_data.get("audio_filename", "")}
            )

        chunks = []
        current_chunk = {
            "text": "",
            "start": None,
            "end": None,
            "segment_ids": []
        }

        for segment in segments:
            segment_text = segment.get("text", "")
            segment_id = segment.get("id", len(current_chunk["segment_ids"]))

            # Neu them segment nay vuot qua chunk_size
            if len(current_chunk["text"]) + len(segment_text) > self.chunk_size and current_chunk["text"]:
                # Luu chunk hien tai
                chunks.append(current_chunk.copy())

                # Tao chunk moi voi overlap
                overlap_text = current_chunk["text"][-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = {
                    "text": overlap_text + " " + segment_text if overlap_text else segment_text,
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "segment_ids": [segment_id]
                }
            else:
                # Them segment vao chunk hien tai
                if current_chunk["start"] is None:
                    current_chunk["start"] = segment.get("start")

                current_chunk["text"] += " " + segment_text if current_chunk["text"] else segment_text
                current_chunk["end"] = segment.get("end")
                current_chunk["segment_ids"].append(segment_id)

        # Them chunk cuoi cung
        if current_chunk["text"]:
            chunks.append(current_chunk)

        # Them metadata
        chunks_with_metadata = []
        audio_file = transcript_data.get("audio_filename", "")

        for idx, chunk in enumerate(chunks):
            start_time = chunk.get("start")
            end_time = chunk.get("end")

            chunk_data = {
                "chunk_id": idx,
                "text": chunk["text"].strip(),
                "start_time": start_time,
                "end_time": end_time,
                "duration": (end_time - start_time) if (end_time and start_time) else 0,
                "segment_ids": chunk["segment_ids"],
                "audio_file": audio_file,
                "char_count": len(chunk["text"]),
                "word_count": len(chunk["text"].split())
            }
            chunks_with_metadata.append(chunk_data)

        return chunks_with_metadata

    def save_chunks(
        self,
        chunks: List[Dict],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Luu chunks ra file JSON

        Args:
            chunks: List cac chunks
            output_path: Duong dan file output

        Returns:
            Path cua file da luu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Da luu {len(chunks)} chunks tai: {output_path}")
        return output_path


# Test function
if __name__ == "__main__":
    # Test basic chunking (khong can API key)
    print("Testing basic chunking methods...")

    chunker = TextChunker(
        chunk_size=200,
        chunk_overlap=20,
        method="recursive"  # Su dung recursive thay vi semantic de test khong can API
    )

    # Test voi text don gian
    sample_text = """
    Day la doan van ban mau. No co nhieu cau.
    Chung ta se chia no thanh cac chunks.

    Day la paragraph thu hai. No cung co nhieu thong tin.
    Chunking giup chung ta to chuc van ban tot hon.

    Paragraph thu ba noi ve AI va Machine Learning.
    Cac mo hinh hoc sau dang phat trien nhanh chong.
    """

    chunks = chunker.chunk_text(sample_text)
    print(f"\nSo chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Text: {chunk['text'][:50]}...")
        print(f"  Chars: {chunk['char_count']}, Words: {chunk['word_count']}")

    # Test voi transcript data
    print("\n\nTesting transcript chunking...")
    sample_transcript = {
        "audio_filename": "test.mp3",
        "full_text": "Day la toan bo noi dung transcript.",
        "segments": [
            {"id": 0, "text": "Day la cau thu nhat.", "start": 0.0, "end": 2.5},
            {"id": 1, "text": "Day la cau thu hai.", "start": 2.5, "end": 5.0},
            {"id": 2, "text": "Day la cau thu ba rat dai va co nhieu thong tin.", "start": 5.0, "end": 10.0},
        ]
    }

    transcript_chunks = chunker.chunk_transcript(sample_transcript)
    print(f"\nSo transcript chunks: {len(transcript_chunks)}")
    for chunk in transcript_chunks:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Time: {chunk.get('start_time', 'N/A')} - {chunk.get('end_time', 'N/A')}")
        print(f"  Text: {chunk['text']}")

    print("\nChunking Module initialized successfully!")
