"""
Main Pipeline - Hệ thống Truy xuất Thông tin Đa phương thức từ Âm thanh
Kết hợp ASR, Chunking, Embedding, Vector DB và RAG với LLM
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from modules import WhisperASR, TextChunker, TextEmbedding, VectorDatabase, RAGSystem
import argparse
import json
from typing import List, Optional


class AudioIRPipeline:
    """
    Pipeline chính cho hệ thống IR từ audio
    """

    def __init__(
        self,
        whisper_model: str = None,
        embedding_model: str = None,
        vector_db_type: str = None,
        llm_model: str = None
    ):
        """
        Khởi tạo pipeline

        Args:
            whisper_model: Tên model Whisper
            embedding_model: Tên model embedding
            vector_db_type: Loại vector database
            llm_model: Tên model LLM
        """
        # Use config values or override
        self.whisper_model = whisper_model or Config.WHISPER_MODEL
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.vector_db_type = vector_db_type or Config.VECTOR_DB_TYPE
        self.llm_model = llm_model or Config.LLM_MODEL

        print("=" * 80)
        print("KHỞI TẠO HỆ THỐNG AUDIO INFORMATION RETRIEVAL")
        print("=" * 80)

        # Initialize components
        self._init_components()

        print("\n✓ Hệ thống đã sẵn sàng!")
        print("=" * 80)

    def _init_components(self):
        """Khởi tạo các components của hệ thống"""

        # 1. ASR Module
        print("\n[1/5] Khởi tạo ASR Module...")
        self.asr = WhisperASR(
            model_name=self.whisper_model,
            device=Config.WHISPER_DEVICE
        )

        # 2. Chunking Module
        print("\n[2/5] Khởi tạo Chunking Module...")
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            method=Config.CHUNKING_METHOD
        )

        # 3. Embedding Module
        print("\n[3/5] Khởi tạo Embedding Module...")
        self.embedder = TextEmbedding(
            model_name=self.embedding_model
        )

        # 4. Vector Database
        print("\n[4/5] Khởi tạo Vector Database...")
        self.vector_db = VectorDatabase(
            db_type=self.vector_db_type,
            db_path=Config.VECTOR_DB_DIR,
            collection_name=Config.COLLECTION_NAME,
            embedding_dimension=Config.EMBEDDING_DIMENSION
        )

        # 5. RAG System
        print("\n[5/5] Khởi tạo RAG System...")
        self.rag = RAGSystem(
            vector_db=self.vector_db,
            embedder=self.embedder,
            llm_model=self.llm_model,
            api_key=Config.OPENAI_API_KEY,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            top_k=Config.TOP_K
        )

    def process_audio(
        self,
        audio_path: str,
        save_intermediate: bool = True
    ) -> dict:
        """
        Xử lý một file audio: ASR -> Chunking -> Embedding -> Store

        Args:
            audio_path: Đường dẫn file audio
            save_intermediate: Có lưu kết quả trung gian không

        Returns:
            Dict chứa thông tin xử lý
        """
        audio_path = Path(audio_path)
        print(f"\n{'=' * 80}")
        print(f"XỬ LÝ AUDIO: {audio_path.name}")
        print(f"{'=' * 80}")

        # Step 1: ASR
        print("\n[Step 1/4] Transcribing audio...")
        transcript_data = self.asr.transcribe_audio(audio_path)

        if save_intermediate:
            transcript_file = Config.TRANSCRIPT_DIR / f"{audio_path.stem}_transcript.json"
            self.asr.save_transcript(transcript_data, transcript_file)

        # Step 2: Chunking
        print("\n[Step 2/4] Chunking transcript...")
        chunks = self.chunker.chunk_transcript(
            transcript_data,
            preserve_timestamps=True
        )
        print(f"✓ Đã tạo {len(chunks)} chunks")

        # Step 3: Embedding
        print("\n[Step 3/4] Creating embeddings...")
        chunks_with_embeddings = self.embedder.encode_chunks(chunks)

        # Step 4: Store in Vector DB
        print("\n[Step 4/4] Storing in vector database...")
        num_stored = self.vector_db.add_documents(chunks_with_embeddings)

        print(f"\n✓ Hoàn thành! Đã xử lý và lưu {num_stored} chunks từ {audio_path.name}")

        return {
            "audio_file": str(audio_path),
            "num_chunks": len(chunks),
            "num_stored": num_stored,
            "transcript_data": transcript_data
        }

    def process_audio_batch(
        self,
        audio_files: List[str],
        save_intermediate: bool = True
    ) -> List[dict]:
        """
        Xử lý nhiều file audio

        Args:
            audio_files: List đường dẫn các file audio
            save_intermediate: Có lưu kết quả trung gian không

        Returns:
            List các kết quả xử lý
        """
        results = []

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n\n{'#' * 80}")
            print(f"FILE {i}/{len(audio_files)}")
            print(f"{'#' * 80}")

            try:
                result = self.process_audio(audio_file, save_intermediate)
                results.append(result)
            except Exception as e:
                print(f"✗ Lỗi khi xử lý {audio_file}: {str(e)}")
                results.append({
                    "audio_file": str(audio_file),
                    "error": str(e)
                })

        return results

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        verbose: bool = True
    ) -> dict:
        """
        Thực hiện query trên hệ thống

        Args:
            question: Câu hỏi
            top_k: Số lượng chunks retrieve
            verbose: Hiển thị chi tiết

        Returns:
            Response từ RAG system
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"QUERY: {question}")
            print(f"{'=' * 80}")

        response = self.rag.query(
            question=question,
            top_k=top_k,
            return_sources=True
        )

        if verbose:
            self._print_response(response)

        return response

    def _print_response(self, response: dict):
        """In response một cách đẹp mắt"""
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(response["answer"])

        if "sources" in response and response["sources"]:
            print("\n" + "=" * 80)
            print(f"SOURCES ({len(response['sources'])} chunks):")
            print("=" * 80)

            for i, source in enumerate(response["sources"], 1):
                print(f"\n[Source {i}] Similarity: {source['similarity']:.4f}")

                if "audio_file" in source and source["audio_file"]:
                    print(f"Audio: {source['audio_file']}")

                if "start_time_formatted" in source:
                    print(f"Time: {source['start_time_formatted']} - {source['end_time_formatted']}")

                print(f"Text: {source['text'][:200]}...")

    def get_stats(self) -> dict:
        """Lấy thống kê về hệ thống"""
        stats = self.vector_db.get_collection_stats()
        return stats

    def interactive_mode(self):
        """Chế độ tương tác với người dùng"""
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE - Nhập câu hỏi để truy vấn")
        print("Gõ 'exit' hoặc 'quit' để thoát")
        print("Gõ 'stats' để xem thống kê")
        print("=" * 80)

        while True:
            try:
                question = input("\n💬 Câu hỏi của bạn: ").strip()

                if not question:
                    continue

                if question.lower() in ["exit", "quit", "q"]:
                    print("Tạm biệt!")
                    break

                if question.lower() == "stats":
                    stats = self.get_stats()
                    print(json.dumps(stats, indent=2, ensure_ascii=False))
                    continue

                # Query
                self.query(question)

            except KeyboardInterrupt:
                print("\n\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Hệ thống Truy xuất Thông tin Đa phương thức từ Âm thanh"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["process", "query", "interactive"],
        default="interactive",
        help="Chế độ hoạt động: process (xử lý audio), query (truy vấn), interactive (tương tác)"
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Đường dẫn file audio hoặc thư mục chứa audio files"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Câu hỏi để query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Số lượng chunks retrieve"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AudioIRPipeline()

    # Execute based on mode
    if args.mode == "process":
        if not args.audio:
            print("Lỗi: Cần chỉ định --audio để xử lý")
            return

        audio_path = Path(args.audio)

        if audio_path.is_file():
            # Process single file
            pipeline.process_audio(str(audio_path))
        elif audio_path.is_dir():
            # Process all audio files in directory
            audio_files = list(audio_path.glob("*.mp3")) + \
                         list(audio_path.glob("*.wav")) + \
                         list(audio_path.glob("*.m4a"))

            if not audio_files:
                print(f"Không tìm thấy file audio trong {audio_path}")
                return

            pipeline.process_audio_batch([str(f) for f in audio_files])
        else:
            print(f"Không tìm thấy: {audio_path}")

    elif args.mode == "query":
        if not args.question:
            print("Lỗi: Cần chỉ định --question để query")
            return

        pipeline.query(args.question, top_k=args.top_k)

    else:  # interactive
        pipeline.interactive_mode()


if __name__ == "__main__":
    main()
