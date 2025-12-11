"""
Streamlit Web UI cho Audio Information Retrieval System
"""

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config

# Page config
st.set_page_config(
    page_title="Audio IR System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .timestamp-badge {
        background-color: #1E88E5;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def load_pipeline(embedding_provider, embedding_model, llm_provider, llm_model):
    """Load or reload the pipeline with specified settings"""
    try:
        from main import AudioIRPipeline

        with st.spinner("Dang khoi tao he thong..."):
            pipeline = AudioIRPipeline(
                embedding_provider=embedding_provider,
                llm_provider=llm_provider
            )
        return pipeline
    except Exception as e:
        st.error(f"Loi khoi tao: {str(e)}")
        return None


def process_audio_file(pipeline, audio_file):
    """Process uploaded audio file"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner(f"Dang xu ly: {audio_file.name}..."):
            result = pipeline.process_audio(tmp_path)
        return result
    finally:
        # Cleanup temp file
        os.unlink(tmp_path)


def display_sources(sources):
    """Display source chunks with timestamps"""
    if not sources:
        return

    st.markdown("**Nguon tham khao:**")
    for i, source in enumerate(sources, 1):
        with st.expander(f"Nguon {i} - Similarity: {source.get('similarity', 0):.2%}"):
            # Timestamp info
            start = source.get('start_time_formatted', 'N/A')
            end = source.get('end_time_formatted', 'N/A')
            audio_file = source.get('audio_file', 'N/A')

            col1, col2, col3 = st.columns(3)
            col1.metric("File", audio_file)
            col2.metric("Bat dau", start)
            col3.metric("Ket thuc", end)

            # Text content
            st.markdown("**Noi dung:**")
            st.info(source.get('text', ''))


def sidebar_settings():
    """Render sidebar settings"""
    st.sidebar.markdown("## Cau hinh")

    # Embedding settings
    st.sidebar.markdown("### Embedding")
    embedding_provider = st.sidebar.selectbox(
        "Provider",
        ["local", "google", "openai"],
        index=0,
        help="local: Sentence-BERT/E5 (mien phi), google/openai: Cloud API"
    )

    if embedding_provider == "local":
        embedding_model = st.sidebar.selectbox(
            "Model",
            ["sbert", "e5", "e5-large", "vi-sbert"],
            index=0
        )
    else:
        embedding_model = None

    # LLM settings
    st.sidebar.markdown("### LLM")
    llm_provider = st.sidebar.selectbox(
        "Provider",
        ["ollama", "google", "openai"],
        index=0,
        help="ollama: Local (mien phi), google/openai: Cloud API"
    )

    if llm_provider == "ollama":
        llm_model = st.sidebar.selectbox(
            "Model",
            ["llama3.2", "llama3.2:1b", "qwen2.5", "qwen2.5:3b", "mistral", "gemma2:2b", "phi3"],
            index=0
        )

        # Check Ollama status
        from modules.rag_module import check_ollama_status
        status = check_ollama_status()
        if status['server_running']:
            st.sidebar.success(f"Ollama: Running")
            if status['available_models']:
                st.sidebar.info(f"Models: {', '.join(status['available_models'][:3])}")
        else:
            st.sidebar.error("Ollama: Not running")
            st.sidebar.caption("Chay: `ollama serve`")
    else:
        llm_model = None

    # Retrieval settings
    st.sidebar.markdown("### Retrieval")
    top_k = st.sidebar.slider("So luong chunks (top_k)", 1, 10, 5)

    return {
        'embedding_provider': embedding_provider,
        'embedding_model': embedding_model,
        'llm_provider': llm_provider,
        'llm_model': llm_model,
        'top_k': top_k
    }


def main():
    """Main application"""
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🎵 Audio Information Retrieval</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Tim kiem va hoi dap thong minh tren noi dung audio</p>', unsafe_allow_html=True)

    # Sidebar settings
    settings = sidebar_settings()

    # Initialize/Reload pipeline button
    st.sidebar.markdown("---")
    if st.sidebar.button("Khoi tao/Reload He thong", type="primary"):
        st.session_state.pipeline = load_pipeline(
            settings['embedding_provider'],
            settings['embedding_model'],
            settings['llm_provider'],
            settings['llm_model']
        )
        if st.session_state.pipeline:
            st.sidebar.success("Da khoi tao thanh cong!")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Xu ly", "💬 Hoi dap", "📊 Thong ke"])

    # Tab 1: Upload and Process
    with tab1:
        st.markdown("### Upload file audio")

        uploaded_files = st.file_uploader(
            "Chon file audio (mp3, wav, m4a, flac)",
            type=['mp3', 'wav', 'm4a', 'flac'],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file da chon:**")
            for f in uploaded_files:
                st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

            if st.button("Xu ly tat ca", type="primary"):
                if st.session_state.pipeline is None:
                    st.warning("Vui long khoi tao he thong truoc!")
                else:
                    progress_bar = st.progress(0)
                    for i, audio_file in enumerate(uploaded_files):
                        result = process_audio_file(st.session_state.pipeline, audio_file)
                        if result:
                            st.session_state.processed_files.append(audio_file.name)
                            st.success(f"Da xu ly: {audio_file.name}")
                        progress_bar.progress((i + 1) / len(uploaded_files))

                    st.balloons()

        # Show processed files
        if st.session_state.processed_files:
            st.markdown("### File da xu ly")
            for f in st.session_state.processed_files:
                st.write(f"✅ {f}")

    # Tab 2: Q&A
    with tab2:
        st.markdown("### Hoi dap ve noi dung audio")

        if st.session_state.pipeline is None:
            st.warning("Vui long khoi tao he thong o sidebar truoc!")
        else:
            # Question input
            question = st.text_input(
                "Nhap cau hoi:",
                placeholder="Noi dung chinh cua audio la gi?"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                search_btn = st.button("Tim kiem", type="primary")

            if search_btn and question:
                with st.spinner("Dang tim kiem va tao cau tra loi..."):
                    start_time = time.time()
                    response = st.session_state.pipeline.query(
                        question,
                        top_k=settings['top_k']
                    )
                    elapsed = time.time() - start_time

                # Display answer
                st.markdown("### Tra loi")
                st.success(response.get('answer', 'Khong tim thay cau tra loi'))

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Thoi gian", f"{elapsed:.2f}s")
                col2.metric("So nguon", response.get('num_sources', 0))
                col3.metric("Model", response.get('model', 'N/A'))

                # Sources
                display_sources(response.get('sources', []))

                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response.get('answer', ''),
                    'time': elapsed
                })

            # Chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("### Lich su hoi dap")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                    with st.expander(f"Q: {chat['question'][:50]}..."):
                        st.markdown(f"**Cau hoi:** {chat['question']}")
                        st.markdown(f"**Tra loi:** {chat['answer']}")
                        st.caption(f"Thoi gian: {chat['time']:.2f}s")

    # Tab 3: Statistics
    with tab3:
        st.markdown("### Thong ke he thong")

        if st.session_state.pipeline:
            try:
                stats = st.session_state.pipeline.get_stats()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tong chunks", stats.get('total_chunks', 0))
                col2.metric("Tong audio files", stats.get('total_audio_files', 0))
                col3.metric("Embedding model", stats.get('embedding_model', 'N/A')[:20])
                col4.metric("LLM model", stats.get('llm_model', 'N/A')[:20])

                # Vector DB stats
                st.markdown("### Vector Database")
                db_stats = stats.get('vector_db_stats', {})
                st.json(db_stats)

            except Exception as e:
                st.error(f"Loi lay thong ke: {e}")
        else:
            st.info("Khoi tao he thong de xem thong ke")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Huong dan")
    st.sidebar.markdown("""
    1. Cau hinh Embedding & LLM
    2. Click **Khoi tao He thong**
    3. Upload file audio
    4. Dat cau hoi
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Audio IR System v1.0")


if __name__ == "__main__":
    main()
