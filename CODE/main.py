"""
Main CLI - Multimodal Information Retrieval System
===================================================

CLI interface for processing documents, audio, video and querying with RAG.
Supports 34 file formats with anti-hallucination and TTS output.

Usage:
    python main.py --mode process --input data/documents/
    python main.py --mode query --question "What is...?"
    python main.py --mode interactive
    python main.py --mode stats
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import json
from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

from config import Config


class MultimodalIRPipeline:
    """
    Main pipeline for Multimodal Information Retrieval system.

    Supports:
    - 34 file formats (documents, audio, video, images, code)
    - Local (SBERT/E5) and Cloud (OpenAI/Google) embeddings
    - Hybrid search (Vector + BM25) with optional reranking
    - Anti-hallucination (verification, conflict detection, abstention)
    - Text-to-Speech output
    """

    def __init__(
        self,
        embedding_provider: str = None,
        llm_provider: str = None,
        use_reranker: bool = False,
        use_anti_hallucination: bool = True,
        verbose: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            embedding_provider: local, google, or openai
            llm_provider: ollama, google, or openai
            use_reranker: Enable cross-encoder reranking
            use_anti_hallucination: Enable answer verification
            verbose: Print initialization progress
        """
        self.embedding_provider = embedding_provider or Config.EMBEDDING_PROVIDER
        self.llm_provider = llm_provider or Config.LLM_PROVIDER
        self.use_reranker = use_reranker
        self.use_anti_hallucination = use_anti_hallucination
        self.verbose = verbose

        if verbose:
            print("=" * 70)
            print("MULTIMODAL INFORMATION RETRIEVAL SYSTEM")
            print(f"Embedding: {self.embedding_provider.upper()} | LLM: {self.llm_provider.upper()}")
            print("=" * 70)

        self._init_components()

        if verbose:
            print("\nSystem ready!")
            print("=" * 70)

    def _init_components(self):
        """Initialize all system components."""
        from modules import (
            TextEmbedding, VectorDatabase, RAGSystem,
            UnifiedProcessor, TextChunker, KnowledgeBase,
            PromptTemplateManager
        )

        # 1. Document Processor
        if self.verbose:
            print("\n[1/6] Initializing Document Processor (34 formats)...")
        self.processor = UnifiedProcessor()

        # 2. Text Chunker
        if self.verbose:
            print("[2/6] Initializing Text Chunker...")
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            method=Config.CHUNKING_METHOD
        )

        # 3. Embedding
        if self.verbose:
            print(f"[3/6] Initializing Embedder ({self.embedding_provider})...")
        self.embedder = TextEmbedding(
            provider=self.embedding_provider,
            model_name=self._get_embedding_model()
        )

        # 4. Vector Database
        if self.verbose:
            print("[4/6] Initializing Vector Database (Qdrant + BM25)...")
        self.vector_db = VectorDatabase(
            collection_name=Config.COLLECTION_NAME,
            embedding_dimension=self.embedder.embedding_dim
        )

        # 5. RAG System
        if self.verbose:
            print(f"[5/6] Initializing RAG System ({self.llm_provider})...")
        self.rag = RAGSystem(
            vector_db=self.vector_db,
            embedder=self.embedder,
            provider=self.llm_provider,
            top_k=Config.TOP_K,
            temperature=Config.LLM_TEMPERATURE
        )

        # 6. Prompt Templates
        if self.verbose:
            print("[6/6] Loading Prompt Templates...")
        self.prompt_manager = PromptTemplateManager(language="vi")

        # Optional: Reranker
        self.reranker = None
        if self.use_reranker:
            try:
                from modules import CrossEncoderReranker
                self.reranker = CrossEncoderReranker()
                if self.verbose:
                    print("[+] Reranker enabled")
            except Exception as e:
                if self.verbose:
                    print(f"[!] Reranker not available: {e}")

        # Optional: Anti-hallucination
        self.verifier = None
        self.abstention_checker = None
        self.conflict_detector = None
        if self.use_anti_hallucination:
            try:
                from modules import AnswerVerifier, AbstentionChecker, ConflictDetector
                self.verifier = AnswerVerifier()
                self.abstention_checker = AbstentionChecker(min_retrieval_score=0.4)
                self.conflict_detector = ConflictDetector()
                if self.verbose:
                    print("[+] Anti-hallucination enabled")
            except Exception as e:
                if self.verbose:
                    print(f"[!] Anti-hallucination not available: {e}")

        # Optional: TTS
        self.tts = None
        try:
            from modules import TextToSpeech
            self.tts = TextToSpeech(voice="vi-female")
            if self.verbose:
                print("[+] TTS enabled")
        except Exception as e:
            if self.verbose:
                print(f"[!] TTS not available: {e}")

    def _get_embedding_model(self) -> str:
        """Get embedding model name based on provider."""
        if self.embedding_provider == "local":
            return getattr(Config, 'LOCAL_EMBEDDING_MODEL', 'e5')
        elif self.embedding_provider == "google":
            return Config.GOOGLE_EMBEDDING_MODEL
        else:
            return Config.OPENAI_EMBEDDING_MODEL

    def process_file(self, file_path: str, tags: List[str] = None) -> dict:
        """
        Process a single file: Extract -> Chunk -> Embed -> Store.

        Args:
            file_path: Path to the file
            tags: Optional tags for the document

        Returns:
            Processing result dict
        """
        file_path = Path(file_path)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"PROCESSING: {file_path.name}")
            print(f"{'=' * 70}")

        # Step 1: Extract content
        if self.verbose:
            print("\n[1/4] Extracting content...")

        try:
            result = self.processor.process(str(file_path))
            if not result or not result.content:
                return {"file": str(file_path), "error": "No content extracted"}
        except Exception as e:
            return {"file": str(file_path), "error": str(e)}

        # Step 2: Chunk
        if self.verbose:
            print("[2/4] Chunking text...")

        metadata = {
            "source": file_path.name,
            "file_type": file_path.suffix.lower(),
            "processed_at": datetime.now().isoformat()
        }
        if tags:
            metadata["tags"] = tags
        if hasattr(result, 'metadata') and result.metadata:
            metadata.update(result.metadata.extra or {})

        chunks = self.chunker.chunk_text(result.content, metadata=metadata)

        if self.verbose:
            print(f"    Created {len(chunks)} chunks")

        # Step 3: Embed
        if self.verbose:
            print("[3/4] Creating embeddings...")

        for chunk in chunks:
            embedding = self.embedder.encode_text(chunk["text"])
            chunk["embedding"] = embedding.tolist()

        # Step 4: Store
        if self.verbose:
            print("[4/4] Storing in vector database...")

        self.vector_db.add_documents(chunks)

        if self.verbose:
            print(f"\nDone! Processed {len(chunks)} chunks from {file_path.name}")

        return {
            "file": str(file_path),
            "num_chunks": len(chunks),
            "content_length": len(result.content),
            "file_type": file_path.suffix.lower()
        }

    def process_directory(self, dir_path: str, tags: List[str] = None) -> List[dict]:
        """
        Process all supported files in a directory.

        Args:
            dir_path: Path to directory
            tags: Optional tags for all documents

        Returns:
            List of processing results
        """
        dir_path = Path(dir_path)
        supported = self.processor.supported_extensions()

        files = []
        for ext in supported:
            files.extend(dir_path.glob(f"*{ext}"))

        if not files:
            print(f"No supported files found in {dir_path}")
            return []

        print(f"\nFound {len(files)} files to process")

        results = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] ", end="")
            result = self.process_file(str(file_path), tags)
            results.append(result)

        # Summary
        success = sum(1 for r in results if "error" not in r)
        print(f"\n{'=' * 70}")
        print(f"SUMMARY: {success}/{len(files)} files processed successfully")
        print(f"{'=' * 70}")

        return results

    def query(
        self,
        question: str,
        top_k: int = None,
        use_hybrid: bool = True,
        alpha: float = 0.7,
        with_tts: bool = False,
        template: str = "strict_qa"
    ) -> dict:
        """
        Query the system with anti-hallucination checks.

        Args:
            question: The question to ask
            top_k: Number of chunks to retrieve
            use_hybrid: Use hybrid search (vector + BM25)
            alpha: Weight for vector search in hybrid
            with_tts: Generate TTS audio for answer
            template: Prompt template to use

        Returns:
            Response dict with answer, sources, verification
        """
        top_k = top_k or Config.TOP_K

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"QUERY: {question}")
            print(f"{'=' * 70}")

        # Step 1: Retrieve
        if self.verbose:
            print("\n[1/4] Retrieving relevant chunks...")

        query_embedding = self.embedder.encode_query(question)

        if use_hybrid:
            if self.reranker:
                results = self.vector_db.search_with_rerank(
                    query=question,
                    query_embedding=query_embedding,
                    reranker=self.reranker,
                    top_k=top_k
                )
            else:
                results = self.vector_db.hybrid_search(
                    query=question,
                    query_embedding=query_embedding,
                    alpha=alpha,
                    top_k=top_k
                )
        else:
            results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k
            )

        if self.verbose:
            print(f"    Found {len(results)} relevant chunks")

        # Step 2: Check abstention
        should_abstain = False
        abstain_reason = None

        if self.abstention_checker and results:
            contexts = [{"similarity": r.get("similarity", 0), "text": r.get("text", "")} for r in results]
            should_abstain, abstain_reason = self.abstention_checker.should_abstain(question, contexts)

            if should_abstain and self.verbose:
                print(f"    [!] Low confidence: {abstain_reason}")

        # Step 3: Check conflicts
        conflict_result = None
        if self.conflict_detector and results and not should_abstain:
            chunks_for_conflict = [
                {
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", {}),
                    "similarity": r.get("similarity", 0)
                }
                for r in results
            ]
            conflict_result = self.conflict_detector.detect_and_resolve(chunks_for_conflict, question)

            if conflict_result.has_conflicts and self.verbose:
                print(f"    [!] Conflicts detected: {conflict_result.conflict_summary}")

        # Step 4: Generate answer
        if self.verbose:
            print("[2/4] Generating answer...")

        if should_abstain:
            answer = f"Xin loi, toi khong the tra loi cau hoi nay vi {abstain_reason}"
            verification_result = None
        else:
            # Use resolved context if conflicts detected
            if conflict_result and conflict_result.resolved_context:
                context = conflict_result.resolved_context
            else:
                context = "\n\n".join([r.get("text", "") for r in results])

            # Get prompt
            sys_prompt, user_prompt = self.prompt_manager.format_prompt(
                template,
                context=context,
                question=question
            )

            # Generate
            response = self.rag.query(question, top_k=top_k)
            answer = response.get("answer", "")

            # Step 5: Verify answer
            verification_result = None
            if self.verifier and not should_abstain:
                if self.verbose:
                    print("[3/4] Verifying answer...")
                verification_result = self.verifier.verify(answer, context, question)

                if self.verbose:
                    print(f"    Grounding: {verification_result.grounding_level.value}")
                    print(f"    Confidence: {verification_result.confidence_score:.2f}")

        # Step 6: TTS (optional)
        audio_bytes = None
        if with_tts and self.tts and answer:
            if self.verbose:
                print("[4/4] Generating speech...")
            try:
                audio_bytes = self.tts.synthesize_sync(answer)
            except Exception as e:
                if self.verbose:
                    print(f"    TTS error: {e}")

        # Build response
        response = {
            "question": question,
            "answer": answer,
            "sources": results,
            "should_abstain": should_abstain,
            "abstain_reason": abstain_reason
        }

        if verification_result:
            response["verification"] = {
                "grounding_level": verification_result.grounding_level.value,
                "confidence_score": verification_result.confidence_score,
                "explanation": verification_result.explanation
            }

        if conflict_result and conflict_result.has_conflicts:
            response["conflicts"] = {
                "has_conflicts": True,
                "summary": conflict_result.conflict_summary,
                "resolution": conflict_result.resolution_note
            }

        if audio_bytes:
            response["audio"] = audio_bytes

        # Print response
        if self.verbose:
            self._print_response(response)

        return response

    def _print_response(self, response: dict):
        """Print response in a formatted way."""
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(response["answer"])

        if response.get("should_abstain"):
            print(f"\n[ABSTAINED] {response.get('abstain_reason')}")

        if "verification" in response:
            v = response["verification"]
            print(f"\n[VERIFICATION] {v['grounding_level']} (confidence: {v['confidence_score']:.2f})")

        if "conflicts" in response:
            c = response["conflicts"]
            print(f"\n[CONFLICTS] {c['summary']}")
            print(f"Resolution: {c['resolution']}")

        if response.get("sources"):
            print("\n" + "-" * 70)
            print(f"SOURCES ({len(response['sources'])} chunks):")
            for i, src in enumerate(response["sources"][:3], 1):
                sim = src.get("similarity", 0)
                text = src.get("text", "")[:150]
                source = src.get("metadata", {}).get("source", "unknown")
                print(f"\n[{i}] {source} (sim: {sim:.3f})")
                print(f"    {text}...")

    def get_stats(self) -> dict:
        """Get system statistics."""
        db_stats = self.vector_db.get_collection_stats()

        return {
            "database": db_stats,
            "config": {
                "embedding_provider": self.embedding_provider,
                "embedding_model": self._get_embedding_model(),
                "embedding_dim": self.embedder.embedding_dim,
                "llm_provider": self.llm_provider,
                "chunk_size": Config.CHUNK_SIZE,
                "top_k": Config.TOP_K
            },
            "features": {
                "reranker": self.reranker is not None,
                "anti_hallucination": self.verifier is not None,
                "tts": self.tts is not None,
                "supported_formats": len(self.processor.supported_extensions())
            }
        }

    def interactive_mode(self):
        """Interactive query mode."""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("Commands:")
        print("  exit/quit  - Exit")
        print("  stats      - Show statistics")
        print("  tts on/off - Toggle TTS")
        print("  hybrid on/off - Toggle hybrid search")
        print("=" * 70)

        use_tts = False
        use_hybrid = True

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if not question:
                    continue

                if question.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break

                if question.lower() == "stats":
                    stats = self.get_stats()
                    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
                    continue

                if question.lower() == "tts on":
                    use_tts = True
                    print("TTS: ON")
                    continue
                elif question.lower() == "tts off":
                    use_tts = False
                    print("TTS: OFF")
                    continue

                if question.lower() == "hybrid on":
                    use_hybrid = True
                    print("Hybrid search: ON")
                    continue
                elif question.lower() == "hybrid off":
                    use_hybrid = False
                    print("Hybrid search: OFF")
                    continue

                # Query
                self.query(question, use_hybrid=use_hybrid, with_tts=use_tts)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal Information Retrieval System - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode process --input data/documents/
  python main.py --mode process --input data/audio/lecture.mp3
  python main.py --mode query --question "What is machine learning?"
  python main.py --mode query --question "AI la gi?" --tts
  python main.py --mode interactive
  python main.py --mode stats
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["process", "query", "interactive", "stats"],
        default="interactive",
        help="Operation mode"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input file or directory path (for process mode)"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Question to query (for query mode)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )

    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable TTS for answer"
    )

    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search"
    )

    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable cross-encoder reranking"
    )

    parser.add_argument(
        "--no-anti-hallucination",
        action="store_true",
        help="Disable anti-hallucination checks"
    )

    parser.add_argument(
        "--embedding",
        type=str,
        choices=["local", "google", "openai"],
        help="Embedding provider"
    )

    parser.add_argument(
        "--llm",
        type=str,
        choices=["ollama", "google", "openai"],
        help="LLM provider"
    )

    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Tags for processed documents"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = MultimodalIRPipeline(
        embedding_provider=args.embedding,
        llm_provider=args.llm,
        use_reranker=args.reranker,
        use_anti_hallucination=not args.no_anti_hallucination,
        verbose=not args.quiet
    )

    # Execute based on mode
    if args.mode == "process":
        if not args.input:
            print("Error: --input required for process mode")
            return 1

        input_path = Path(args.input)

        if input_path.is_file():
            result = pipeline.process_file(str(input_path), tags=args.tags)
            if "error" in result:
                print(f"Error: {result['error']}")
                return 1
        elif input_path.is_dir():
            results = pipeline.process_directory(str(input_path), tags=args.tags)
            errors = [r for r in results if "error" in r]
            if errors:
                print(f"\n{len(errors)} files failed to process")
                return 1
        else:
            print(f"Error: Path not found: {input_path}")
            return 1

    elif args.mode == "query":
        if not args.question:
            print("Error: --question required for query mode")
            return 1

        pipeline.query(
            args.question,
            top_k=args.top_k,
            use_hybrid=not args.no_hybrid,
            with_tts=args.tts
        )

    elif args.mode == "stats":
        stats = pipeline.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))

    else:  # interactive
        pipeline.interactive_mode()

    return 0


if __name__ == "__main__":
    sys.exit(main())
