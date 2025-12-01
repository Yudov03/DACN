"""
Chunking Module - Phân đoạn văn bản theo ngữ nghĩa
Chia văn bản thành các chunks hợp lý để embedding và retrieval
"""

import re
from typing import List, Dict, Optional, Union
from pathlib import Path
import json


class TextChunker:
    """
    Lớp xử lý chunking văn bản với nhiều phương pháp khác nhau
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        method: str = "semantic"
    ):
        """
        Khởi tạo Text Chunker

        Args:
            chunk_size: Kích thước chunk (số ký tự)
            chunk_overlap: Số ký tự overlap giữa các chunks
            method: Phương pháp chunking (fixed, semantic, sentence)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method

        if self.method not in ["fixed", "semantic", "sentence"]:
            raise ValueError(f"Method '{method}' không hợp lệ. Chọn: fixed, semantic, sentence")

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chia văn bản thành các chunks

        Args:
            text: Văn bản cần chunk
            metadata: Metadata bổ sung cho chunks

        Returns:
            List các chunks với metadata
        """
        if self.method == "fixed":
            chunks = self._fixed_size_chunking(text)
        elif self.method == "sentence":
            chunks = self._sentence_based_chunking(text)
        else:  # semantic
            chunks = self._semantic_chunking(text)

        # Thêm metadata
        return self._add_metadata(chunks, metadata or {})

    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Chunking với kích thước cố định

        Args:
            text: Văn bản cần chunk

        Returns:
            List các chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Tránh cắt giữa từ
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    end = start + last_space

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return chunks

    def _sentence_based_chunking(self, text: str) -> List[str]:
        """
        Chunking dựa trên câu (sentence boundaries)

        Args:
            text: Văn bản cần chunk

        Returns:
            List các chunks
        """
        # Tách thành các câu
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Nếu thêm câu này vào chunk hiện tại vượt quá chunk_size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Giữ overlap
                current_chunk = current_chunk[-self.chunk_overlap:] + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Chunking dựa trên ngữ nghĩa (paragraph + sentence boundaries)

        Args:
            text: Văn bản cần chunk

        Returns:
            List các chunks
        """
        # Tách thành paragraphs
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Nếu paragraph nhỏ hơn chunk_size, thử gộp với chunk hiện tại
            if len(para) <= self.chunk_size:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
            else:
                # Paragraph quá lớn, chia nhỏ theo câu
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Chia paragraph theo câu
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence

        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _add_metadata(self, chunks: List[str], base_metadata: Dict) -> List[Dict]:
        """
        Thêm metadata cho các chunks

        Args:
            chunks: List các chunks
            base_metadata: Metadata cơ bản

        Returns:
            List các chunks với metadata
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
        Chunk transcript từ ASR output, giữ lại timestamps

        Args:
            transcript_data: Dữ liệu transcript từ ASR
            preserve_timestamps: Có giữ timestamps không

        Returns:
            List các chunks với timestamps
        """
        if preserve_timestamps and "segments" in transcript_data:
            return self._chunk_with_timestamps(transcript_data)
        else:
            # Chunk toàn bộ full_text
            return self.chunk_text(
                transcript_data["full_text"],
                metadata={
                    "audio_file": transcript_data.get("audio_filename", ""),
                    "source": "transcript"
                }
            )

    def _chunk_with_timestamps(self, transcript_data: Dict) -> List[Dict]:
        """
        Chunk transcript giữ nguyên timestamps từ các segments

        Args:
            transcript_data: Dữ liệu transcript

        Returns:
            List các chunks với timestamps
        """
        segments = transcript_data["segments"]
        chunks = []

        current_chunk = {
            "text": "",
            "start": None,
            "end": None,
            "segment_ids": []
        }

        for segment in segments:
            segment_text = segment["text"]

            # Nếu thêm segment này vượt quá chunk_size
            if len(current_chunk["text"]) + len(segment_text) > self.chunk_size and current_chunk["text"]:
                # Lưu chunk hiện tại
                chunks.append(current_chunk.copy())

                # Tạo chunk mới với overlap
                overlap_text = current_chunk["text"][-self.chunk_overlap:]
                current_chunk = {
                    "text": overlap_text + " " + segment_text,
                    "start": segment["start"],
                    "end": segment["end"],
                    "segment_ids": [segment["id"]]
                }
            else:
                # Thêm segment vào chunk hiện tại
                if current_chunk["start"] is None:
                    current_chunk["start"] = segment["start"]

                current_chunk["text"] += " " + segment_text if current_chunk["text"] else segment_text
                current_chunk["end"] = segment["end"]
                current_chunk["segment_ids"].append(segment["id"])

        # Thêm chunk cuối cùng
        if current_chunk["text"]:
            chunks.append(current_chunk)

        # Thêm metadata
        chunks_with_metadata = []
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": idx,
                "text": chunk["text"].strip(),
                "start_time": chunk["start"],
                "end_time": chunk["end"],
                "duration": chunk["end"] - chunk["start"] if chunk["end"] and chunk["start"] else 0,
                "segment_ids": chunk["segment_ids"],
                "audio_file": transcript_data.get("audio_filename", ""),
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
        Lưu chunks ra file JSON

        Args:
            chunks: List các chunks
            output_path: Đường dẫn file output

        Returns:
            Path của file đã lưu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Đã lưu {len(chunks)} chunks tại: {output_path}")
        return output_path


# Test function
if __name__ == "__main__":
    # Example usage
    chunker = TextChunker(chunk_size=500, chunk_overlap=50, method="semantic")

    # Test với text đơn giản
    sample_text = """
    Đây là đoạn văn bản mẫu. Nó có nhiều câu.
    Chúng ta sẽ chia nó thành các chunks.

    Đây là paragraph thứ hai. Nó cũng có nhiều thông tin.
    Chunking giúp chúng ta tổ chức văn bản tốt hơn.
    """

    chunks = chunker.chunk_text(sample_text)
    print(f"Số chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']}: {chunk['text'][:50]}...")

    print("\nChunking Module initialized successfully!")
