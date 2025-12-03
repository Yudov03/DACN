"""
Main Pipeline - He thong Truy xuat Thong tin Da phuong thuc tu Am thanh
Ho tro ca OpenAI va Google (Gemini) - su dung Qdrant + LangChain
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
    Pipeline chinh cho he thong IR tu audio
    Ho tro ca OpenAI va Google - Qdrant + LangChain
    """

    def __init__(
        self,
        whisper_model: str = None,
        embedding_model: str = None,
        llm_model: str = None,
        qdrant_host: str = None,
        qdrant_port: int = None,
        qdrant_url: str = None,
        llm_provider: str = None,
        embedding_provider: str = None
    ):
        """
        Khoi tao pipeline

        Args:
            whisper_model: Ten model Whisper
            embedding_model: Ten model embedding
            llm_model: Ten model LLM
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            qdrant_url: Qdrant Cloud URL
            llm_provider: Provider cho LLM (openai/google)
            embedding_provider: Provider cho embedding (openai/google)
        """
        # Provider config
        self.llm_provider = llm_provider or Config.LLM_PROVIDER
        self.embedding_provider = embedding_provider or Config.EMBEDDING_PROVIDER

        # Use config values or override based on provider
        self.whisper_model = whisper_model or Config.WHISPER_MODEL

        # Get model names based on provider
        if embedding_model:
            self.embedding_model = embedding_model
        elif self.embedding_provider == "google":
            self.embedding_model = Config.GOOGLE_EMBEDDING_MODEL
        else:
            self.embedding_model = Config.OPENAI_EMBEDDING_MODEL

        if llm_model:
            self.llm_model = llm_model
        elif self.llm_provider == "google":
            self.llm_model = Config.GOOGLE_LLM_MODEL
        else:
            self.llm_model = Config.OPENAI_LLM_MODEL

        # Qdrant config
        qdrant_config = Config.get_qdrant_config()
        self.qdrant_host = qdrant_host or qdrant_config.get("host", "localhost")
        self.qdrant_port = qdrant_port or qdrant_config.get("port", 6333)
        self.qdrant_url = qdrant_url or qdrant_config.get("url")

        print("=" * 80)
        print("KHOI TAO HE THONG AUDIO INFORMATION RETRIEVAL")
        print(f"LLM Provider: {self.llm_provider.upper()} | Embedding Provider: {self.embedding_provider.upper()}")
        print("=" * 80)

        # Initialize components
        self._init_components()

        print("\nHe thong da san sang!")
        print("=" * 80)

    def _init_components(self):
        """Khoi tao cac components cua he thong"""

        # Get API key based on provider
        api_key = Config.get_api_key(self.embedding_provider)
        llm_api_key = Config.get_api_key(self.llm_provider)

        # Get embedding dimension based on provider
        if self.embedding_provider == "google":
            embedding_dim = Config.GOOGLE_EMBEDDING_DIMENSION
        else:
            embedding_dim = Config.OPENAI_EMBEDDING_DIMENSION

        # 1. ASR Module
        print("\n[1/5] Khoi tao ASR Module (Whisper)...")
        self.asr = WhisperASR(
            model_name=self.whisper_model,
            device=Config.WHISPER_DEVICE
        )

        # 2. Chunking Module (LangChain)
        print("\n[2/5] Khoi tao Chunking Module (LangChain)...")
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            method=Config.CHUNKING_METHOD,
            api_key=api_key,
            embedding_provider=self.embedding_provider
        )

        # 3. Embedding Module
        print(f"\n[3/5] Khoi tao Embedding Module ({self.embedding_provider.upper()})...")
        self.embedder = TextEmbedding(
            model_name=self.embedding_model,
            provider=self.embedding_provider,
            api_key=api_key,
            cache_dir=str(Config.DATA_DIR / "embedding_cache")
        )

        # 4. Vector Database (Qdrant)
        print("\n[4/5] Khoi tao Vector Database (Qdrant)...")
        self.vector_db = VectorDatabase(
            host=self.qdrant_host,
            port=self.qdrant_port,
            url=self.qdrant_url,
            api_key=Config.QDRANT_API_KEY,
            collection_name=Config.COLLECTION_NAME,
            embedding_dimension=embedding_dim
        )

        # 5. RAG System (LangChain)
        print(f"\n[5/5] Khoi tao RAG System ({self.llm_provider.upper()})...")
        self.rag = RAGSystem(
            vector_db=self.vector_db,
            embedder=self.embedder,
            llm_model=self.llm_model,
            provider=self.llm_provider,
            api_key=llm_api_key,
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
        Xu ly mot file audio: ASR -> Chunking -> Embedding -> Store

        Args:
            audio_path: Duong dan file audio
            save_intermediate: Co luu ket qua trung gian khong

        Returns:
            Dict chua thong tin xu ly
        """
        audio_path = Path(audio_path)
        print(f"\n{'=' * 80}")
        print(f"XU LY AUDIO: {audio_path.name}")
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
        print(f"Da tao {len(chunks)} chunks")

        # Step 3: Embedding
        print("\n[Step 3/4] Creating embeddings (OpenAI)...")
        chunks_with_embeddings = self.embedder.encode_chunks(chunks)

        # Step 4: Store in Qdrant
        print("\n[Step 4/4] Storing in Qdrant...")
        num_stored = self.vector_db.add_documents(chunks_with_embeddings)

        print(f"\nHoan thanh! Da xu ly va luu {num_stored} chunks tu {audio_path.name}")

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
        Xu ly nhieu file audio

        Args:
            audio_files: List duong dan cac file audio
            save_intermediate: Co luu ket qua trung gian khong

        Returns:
            List cac ket qua xu ly
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
                print(f"Loi khi xu ly {audio_file}: {str(e)}")
                results.append({
                    "audio_file": str(audio_file),
                    "error": str(e)
                })

        return results

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        verbose: bool = True,
        use_mmr: bool = False
    ) -> dict:
        """
        Thuc hien query tren he thong

        Args:
            question: Cau hoi
            top_k: So luong chunks retrieve
            verbose: Hien thi chi tiet
            use_mmr: Su dung MMR de tang diversity

        Returns:
            Response tu RAG system
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"QUERY: {question}")
            print(f"{'=' * 80}")

        response = self.rag.query(
            question=question,
            top_k=top_k,
            return_sources=True,
            use_mmr=use_mmr
        )

        if verbose:
            self._print_response(response)

        return response

    def _print_response(self, response: dict):
        """In response mot cach dep mat"""
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
                    print(f"Time: {source['start_time_formatted']} - {source.get('end_time_formatted', 'N/A')}")

                text = source.get('text', '')
                print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")

    def get_stats(self) -> dict:
        """Lay thong ke ve he thong"""
        db_stats = self.vector_db.get_collection_stats()
        rag_stats = self.rag.get_retriever_stats()

        return {
            "database": db_stats,
            "rag": rag_stats,
            "config": {
                "whisper_model": self.whisper_model,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP
            }
        }

    def interactive_mode(self):
        """Che do tuong tac voi nguoi dung"""
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE - Nhap cau hoi de truy van")
        print("Go 'exit' hoac 'quit' de thoat")
        print("Go 'stats' de xem thong ke")
        print("Go 'mmr' de bat/tat MMR mode")
        print("=" * 80)

        use_mmr = False

        while True:
            try:
                question = input("\nCau hoi cua ban: ").strip()

                if not question:
                    continue

                if question.lower() in ["exit", "quit", "q"]:
                    print("Tam biet!")
                    break

                if question.lower() == "stats":
                    stats = self.get_stats()
                    print(json.dumps(stats, indent=2, ensure_ascii=False))
                    continue

                if question.lower() == "mmr":
                    use_mmr = not use_mmr
                    print(f"MMR mode: {'ON' if use_mmr else 'OFF'}")
                    continue

                # Query
                self.query(question, use_mmr=use_mmr)

            except KeyboardInterrupt:
                print("\n\nTam biet!")
                break
            except Exception as e:
                print(f"Loi: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="He thong Truy xuat Thong tin Da phuong thuc tu Am thanh (Qdrant + LangChain)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["process", "query", "interactive"],
        default="interactive",
        help="Che do hoat dong: process (xu ly audio), query (truy van), interactive (tuong tac)"
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Duong dan file audio hoac thu muc chua audio files"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Cau hoi de query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="So luong chunks retrieve"
    )

    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Su dung MMR de tang diversity"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AudioIRPipeline()

    # Execute based on mode
    if args.mode == "process":
        if not args.audio:
            print("Loi: Can chi dinh --audio de xu ly")
            return

        audio_path = Path(args.audio)

        if audio_path.is_file():
            # Process single file
            pipeline.process_audio(str(audio_path))
        elif audio_path.is_dir():
            # Process all audio files in directory
            audio_files = list(audio_path.glob("*.mp3")) + \
                         list(audio_path.glob("*.wav")) + \
                         list(audio_path.glob("*.m4a")) + \
                         list(audio_path.glob("*.flac"))

            if not audio_files:
                print(f"Khong tim thay file audio trong {audio_path}")
                return

            pipeline.process_audio_batch([str(f) for f in audio_files])
        else:
            print(f"Khong tim thay: {audio_path}")

    elif args.mode == "query":
        if not args.question:
            print("Loi: Can chi dinh --question de query")
            return

        pipeline.query(args.question, top_k=args.top_k, use_mmr=args.mmr)

    else:  # interactive
        pipeline.interactive_mode()


if __name__ == "__main__":
    main()
