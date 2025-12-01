"""
Examples sử dụng hệ thống Audio IR
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from modules import WhisperASR, TextChunker, TextEmbedding, VectorDatabase, RAGSystem


def example_1_basic_usage():
    """Example 1: Sử dụng cơ bản - Process audio và query"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Initialize components
    asr = WhisperASR(model_name="base")
    chunker = TextChunker(chunk_size=500, method="semantic")
    embedder = TextEmbedding()
    vector_db = VectorDatabase(db_type="chromadb")

    # Process audio
    audio_file = "data/audio/sample.mp3"  # Replace with your audio file

    print("\n1. Transcribing audio...")
    transcript = asr.transcribe_audio(audio_file)

    print("\n2. Chunking...")
    chunks = chunker.chunk_transcript(transcript)

    print("\n3. Creating embeddings...")
    chunks_with_embeddings = embedder.encode_chunks(chunks)

    print("\n4. Storing in database...")
    vector_db.add_documents(chunks_with_embeddings)

    # Initialize RAG
    rag = RAGSystem(
        vector_db=vector_db,
        embedder=embedder,
        llm_model="gpt-3.5-turbo",
        api_key=Config.OPENAI_API_KEY
    )

    # Query
    print("\n5. Querying...")
    response = rag.query("Nội dung chính là gì?")
    print(f"\nAnswer: {response['answer']}")


def example_2_individual_modules():
    """Example 2: Sử dụng từng module riêng lẻ"""
    print("=" * 80)
    print("EXAMPLE 2: Individual Modules")
    print("=" * 80)

    # ASR Module
    print("\n--- ASR Module ---")
    asr = WhisperASR(model_name="base", language="vi")

    # Transcribe single file
    # transcript = asr.transcribe_audio("audio.mp3")
    # asr.save_transcript(transcript, "output/transcript.json", format="json")

    # Batch transcribe
    # audio_files = ["audio1.mp3", "audio2.mp3"]
    # transcripts = asr.transcribe_batch(audio_files, "output/")

    # Chunking Module
    print("\n--- Chunking Module ---")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50, method="semantic")

    sample_text = """
    Trí tuệ nhân tạo (AI) đang thay đổi thế giới.
    Công nghệ này có ứng dụng trong nhiều lĩnh vực.

    Machine Learning là một nhánh quan trọng của AI.
    Nó cho phép máy tính học từ dữ liệu.
    """

    chunks = chunker.chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"  - Chunk {chunk['chunk_id']}: {chunk['word_count']} words")

    # Embedding Module
    print("\n--- Embedding Module ---")
    embedder = TextEmbedding()

    # Encode single text
    text = "Xin chào, đây là một câu văn mẫu."
    embedding = embedder.encode_text(text, show_progress=False)
    print(f"Embedding shape: {embedding.shape}")

    # Encode multiple texts
    texts = ["Câu 1", "Câu 2", "Câu 3"]
    embeddings = embedder.encode_text(texts, show_progress=False)
    print(f"Batch embeddings shape: {embeddings.shape}")

    # Compute similarity
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 0 and 1: {sim:.4f}")


def example_3_vector_databases():
    """Example 3: So sánh ChromaDB và FAISS"""
    print("=" * 80)
    print("EXAMPLE 3: Vector Databases Comparison")
    print("=" * 80)

    embedder = TextEmbedding()

    # Sample data
    sample_chunks = [
        {"text": "AI đang phát triển rất nhanh.", "chunk_id": 0},
        {"text": "Machine Learning cần nhiều dữ liệu.", "chunk_id": 1},
        {"text": "Deep Learning sử dụng neural networks.", "chunk_id": 2}
    ]

    # Add embeddings
    chunks_with_embeddings = embedder.encode_chunks(sample_chunks)

    # ChromaDB
    print("\n--- ChromaDB ---")
    db_chroma = VectorDatabase(
        db_type="chromadb",
        collection_name="test_chroma",
        db_path="data/vector_db/chroma_test"
    )
    db_chroma.add_documents(chunks_with_embeddings)

    # Search
    query = "Trí tuệ nhân tạo là gì?"
    query_embedding = embedder.encode_text(query, show_progress=False)
    results = db_chroma.search(query_embedding, top_k=2)

    print(f"\nSearch results for: '{query}'")
    for result in results:
        print(f"  - {result['text']} (similarity: {result['similarity']:.4f})")

    # FAISS
    print("\n--- FAISS ---")
    db_faiss = VectorDatabase(
        db_type="faiss",
        collection_name="test_faiss",
        db_path="data/vector_db/faiss_test",
        embedding_dimension=768
    )
    db_faiss.add_documents(chunks_with_embeddings)

    # Search
    results = db_faiss.search(query_embedding, top_k=2)

    print(f"\nSearch results for: '{query}'")
    for result in results:
        print(f"  - {result['text']} (similarity: {result['similarity']:.4f})")


def example_4_custom_rag():
    """Example 4: Custom RAG với prompt template riêng"""
    print("=" * 80)
    print("EXAMPLE 4: Custom RAG")
    print("=" * 80)

    # Setup (giả sử đã có data trong vector DB)
    embedder = TextEmbedding()
    vector_db = VectorDatabase(db_type="chromadb")

    rag = RAGSystem(
        vector_db=vector_db,
        embedder=embedder,
        llm_model="gpt-3.5-turbo"
    )

    # Custom prompt template
    custom_template = """Bạn là một chuyên gia phân tích nội dung audio.

Thông tin từ audio:
{context}

Câu hỏi: {question}

Hãy trả lời một cách chuyên nghiệp, ngắn gọn và chính xác. Nếu có thông tin thời gian, hãy đề cập cụ thể.

Trả lời:"""

    rag.set_prompt_template(custom_template)

    # Query với custom prompt
    # response = rag.query("Nội dung chính là gì?")
    # print(response["answer"])


def example_5_full_pipeline():
    """Example 5: Full pipeline từ đầu đến cuối"""
    print("=" * 80)
    print("EXAMPLE 5: Full Pipeline")
    print("=" * 80)

    from main import AudioIRPipeline

    # Initialize pipeline
    pipeline = AudioIRPipeline(
        whisper_model="base",
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        vector_db_type="chromadb",
        llm_model="gpt-3.5-turbo"
    )

    # Process audio files
    # audio_files = [
    #     "data/audio/lecture1.mp3",
    #     "data/audio/lecture2.mp3"
    # ]
    # results = pipeline.process_audio_batch(audio_files)

    # Query
    questions = [
        "Chủ đề chính là gì?",
        "Những điểm quan trọng được đề cập?",
        "Ai là người nói?"
    ]

    # responses = pipeline.rag.batch_query(questions)

    # for q, r in zip(questions, responses):
    #     print(f"\nQ: {q}")
    #     print(f"A: {r['answer']}")

    # Stats
    stats = pipeline.get_stats()
    print(f"\nDatabase stats: {stats}")


if __name__ == "__main__":
    print("AUDIO IR SYSTEM - EXAMPLES\n")

    # Uncomment để chạy examples
    # example_1_basic_usage()
    example_2_individual_modules()
    # example_3_vector_databases()
    # example_4_custom_rag()
    # example_5_full_pipeline()

    print("\n✓ Examples completed!")
