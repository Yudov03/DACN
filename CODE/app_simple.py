"""
Simple Streamlit Web UI - Demo nhanh khong can full pipeline
Chi can Embedding + Vector DB de test retrieval
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="Audio IR Demo", page_icon="🎵", layout="wide")

st.title("🎵 Audio Information Retrieval - Demo")
st.caption("Demo retrieval voi sample data")


@st.cache_resource
def load_components():
    """Load embedding and vector db"""
    from modules.embedding_module import TextEmbedding
    from modules.vector_db_module import VectorDatabase

    # Initialize with local embedding
    embedder = TextEmbedding(provider="local", model_name="sbert")

    # Create in-memory vector db
    vector_db = VectorDatabase(
        collection_name="demo_collection",
        embedding_dimension=embedder.embedding_dim
    )

    return embedder, vector_db


def add_sample_data(embedder, vector_db):
    """Add sample Vietnamese audio transcript data"""
    sample_chunks = [
        {
            "text": "Hom nay chung ta se tim hieu ve tri tue nhan tao va ung dung cua no trong cuoc song.",
            "audio_file": "lecture_ai.mp3",
            "start_time": 0.0,
            "end_time": 5.0,
            "chunk_id": 0
        },
        {
            "text": "Machine learning la mot nhanh cua AI, cho phep may tinh hoc tu du lieu ma khong can lap trinh cu the.",
            "audio_file": "lecture_ai.mp3",
            "start_time": 5.0,
            "end_time": 12.0,
            "chunk_id": 1
        },
        {
            "text": "Deep learning su dung mang neural nhieu lop de xu ly du lieu phuc tap nhu hinh anh va am thanh.",
            "audio_file": "lecture_ai.mp3",
            "start_time": 12.0,
            "end_time": 20.0,
            "chunk_id": 2
        },
        {
            "text": "Natural Language Processing giup may tinh hieu va xu ly ngon ngu tu nhien cua con nguoi.",
            "audio_file": "lecture_nlp.mp3",
            "start_time": 0.0,
            "end_time": 6.0,
            "chunk_id": 3
        },
        {
            "text": "Cac mo hinh ngon ngu lon nhu GPT va BERT da tao ra dot pha trong xu ly van ban.",
            "audio_file": "lecture_nlp.mp3",
            "start_time": 6.0,
            "end_time": 12.0,
            "chunk_id": 4
        },
        {
            "text": "Whisper la mo hinh ASR cua OpenAI, co the chuyen doi giong noi thanh van ban voi do chinh xac cao.",
            "audio_file": "lecture_asr.mp3",
            "start_time": 0.0,
            "end_time": 7.0,
            "chunk_id": 5
        },
        {
            "text": "Vector database luu tru embedding vectors va cho phep tim kiem tuong dong ngu nghia.",
            "audio_file": "lecture_vectordb.mp3",
            "start_time": 0.0,
            "end_time": 6.0,
            "chunk_id": 6
        },
        {
            "text": "RAG ket hop retrieval va generation de tao cau tra loi dua tren nguon du lieu cu the.",
            "audio_file": "lecture_rag.mp3",
            "start_time": 0.0,
            "end_time": 6.0,
            "chunk_id": 7
        },
    ]

    # Encode chunks
    texts = [c["text"] for c in sample_chunks]
    embeddings = embedder.encode_text(texts, show_progress=False)

    for chunk, emb in zip(sample_chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    # Add to vector db
    vector_db.add_documents(sample_chunks)

    return len(sample_chunks)


def format_time(seconds):
    """Format seconds to MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    # Load components
    with st.spinner("Dang tai mo hinh..."):
        embedder, vector_db = load_components()

    # Sidebar
    st.sidebar.header("Cau hinh")

    # Add sample data button
    if st.sidebar.button("Them du lieu mau"):
        with st.spinner("Dang them du lieu..."):
            count = add_sample_data(embedder, vector_db)
        st.sidebar.success(f"Da them {count} chunks!")

    # Settings
    top_k = st.sidebar.slider("So ket qua (top_k)", 1, 10, 5)

    # Stats
    stats = vector_db.get_collection_stats()
    st.sidebar.metric("Tong chunks", stats.get('count', 0))
    st.sidebar.metric("Embedding dim", embedder.embedding_dim)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Huong dan:**
    1. Click "Them du lieu mau"
    2. Nhap cau hoi
    3. Click "Tim kiem"
    """)

    # Main content
    st.markdown("### Tim kiem ngu nghia")

    # Query input
    query = st.text_input(
        "Nhap cau hoi:",
        placeholder="VD: Machine learning la gi?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("Tim kiem", type="primary")

    if search_btn and query:
        # Encode query
        with st.spinner("Dang tim kiem..."):
            query_embedding = embedder.encode_query(query)

            # Search
            results = vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k
            )

        if results:
            st.markdown(f"### Ket qua ({len(results)} chunks)")

            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                text = result.get('text', '')
                metadata = result.get('metadata', {})

                # Color based on similarity
                if similarity > 0.7:
                    color = "green"
                elif similarity > 0.5:
                    color = "orange"
                else:
                    color = "red"

                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"**{i}. {text}**")

                        # Metadata
                        audio_file = metadata.get('audio_file', 'N/A')
                        start = format_time(metadata.get('start_time', 0))
                        end = format_time(metadata.get('end_time', 0))
                        st.caption(f"📁 {audio_file} | ⏱️ {start} - {end}")

                    with col2:
                        st.metric("Similarity", f"{similarity:.1%}")

                    st.divider()
        else:
            st.warning("Khong tim thay ket qua. Hay them du lieu mau truoc!")

    # Sample queries
    st.markdown("---")
    st.markdown("### Cau hoi mau")
    sample_queries = [
        "AI la gi?",
        "Machine learning hoat dong nhu the nao?",
        "Deep learning xu ly du lieu gi?",
        "NLP dung de lam gi?",
        "Whisper la gi?",
        "Vector database lam gi?",
        "RAG la gi?"
    ]

    cols = st.columns(4)
    for i, sq in enumerate(sample_queries):
        with cols[i % 4]:
            if st.button(sq, key=f"sq_{i}"):
                st.session_state['query'] = sq
                st.rerun()


if __name__ == "__main__":
    main()
