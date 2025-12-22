"""
Student Portal - Hệ thống tra cứu thông tin
===========================================

Giao diện dành cho SINH VIÊN tra cứu thông tin từ Knowledge Base.
Chỉ có chức năng tìm kiếm và hỏi đáp, KHÔNG có upload/quản lý.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
import subprocess
import time
import requests
import warnings
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Voice input
try:
    from audio_recorder_streamlit import audio_recorder
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    print("Warning: audio-recorder-streamlit not installed. Voice input disabled.")

# Suppress PyTorch internal warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Examining the path.*")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Note: Don't use Windows encoding wrapper here - Streamlit manages its own output streams
# Using TextIOWrapper causes "I/O operation on closed file" errors on hot-reload

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Get config from .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_EMBEDDING_DIMENSION = int(os.getenv("LOCAL_EMBEDDING_DIMENSION", 768))


# =============================================================================
# Auto-start Ollama
# =============================================================================

def is_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_ollama():
    """Start Ollama server in background"""
    if is_ollama_running():
        return True

    try:
        # Start Ollama in background (Windows)
        if sys.platform == "win32":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # Wait for server to start (max 10 seconds)
        for _ in range(20):
            time.sleep(0.5)
            if is_ollama_running():
                return True
        return False
    except FileNotFoundError:
        return False  # Ollama not installed
    except Exception:
        return False


# Auto-start Ollama when app loads (only if using Ollama provider)
if LLM_PROVIDER == "ollama" and "ollama_started" not in st.session_state:
    st.session_state.ollama_started = start_ollama()
    if not st.session_state.ollama_started:
        print(f"Warning: Could not start Ollama. Make sure it's installed.")

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Tra cứu thông tin",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    /* Chat styling */
    .stChatMessage {
        padding: 0.5rem;
    }

    /* Source citation */
    .source-box {
        background-color: #E3F2FD;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border-left: 4px solid #1E88E5;
        font-size: 0.85rem;
    }

    /* Stats bar */
    .stats-bar {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
    }

    /* Voice input hint */
    .voice-hint {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background: #f0f7ff;
        border-radius: 5px;
    }

    /* Voice transcript display */
    .voice-transcript {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }

    /* Mobile optimization */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.4rem;
        }
        .main-header p {
            font-size: 0.9rem;
        }
        [data-testid="column"] {
            padding: 0.25rem !important;
        }
    }

    /* Hide unnecessary elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "kb" not in st.session_state:
    st.session_state.kb = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "tts" not in st.session_state:
    st.session_state.tts = None

if "asr" not in st.session_state:
    st.session_state.asr = None

if "voice_query" not in st.session_state:
    st.session_state.voice_query = None

if "auto_tts" not in st.session_state:
    st.session_state.auto_tts = True  # Auto-play TTS for voice queries by default


# =============================================================================
# Initialize Components
# =============================================================================

@st.cache_resource
def init_embedder():
    """Initialize embedding model (cached)"""
    try:
        from src.modules import TextEmbedding
        return TextEmbedding(provider=EMBEDDING_PROVIDER)  # From .env
    except Exception as e:
        st.error(f"Lỗi khởi tạo Embedding: {e}")
        return None


@st.cache_resource
def init_vector_db():
    """Initialize vector database (cached)"""
    try:
        from src.modules import VectorDatabase
        return VectorDatabase(
            collection_name="knowledge_base",
            embedding_dimension=LOCAL_EMBEDDING_DIMENSION  # From .env
        )
    except Exception as e:
        st.error(f"Lỗi kết nối Qdrant: {e}")
        return None


def get_kb_stats():
    """Get knowledge base statistics"""
    try:
        from src.modules import KnowledgeBase
        kb_dir = Path(__file__).parent / "data" / "knowledge_base"
        kb = KnowledgeBase(base_dir=str(kb_dir))
        return kb.get_stats()
    except:
        return None


@st.cache_resource
def init_rag():
    """Initialize RAG system (cached)"""
    try:
        from src.modules import RAGSystem
        embedder = init_embedder()
        vector_db = init_vector_db()

        if embedder and vector_db:
            rag = RAGSystem(
                vector_db=vector_db,
                embedder=embedder,
                provider=LLM_PROVIDER,
                enable_verification=True,
                prompt_template_name="strict_qa"
            )
            return rag
    except Exception as e:
        print(f"Lỗi khởi tạo RAG: {e}")
    return None


def init_tts():
    """Initialize TTS"""
    if st.session_state.tts is None:
        try:
            from src.modules import TextToSpeech
            st.session_state.tts = TextToSpeech(voice="vi-female")
        except:
            pass
    return st.session_state.tts


@st.cache_resource
def init_asr():
    """Initialize ASR (WhisperASR) for voice input"""
    try:
        from src.modules import WhisperASR
        return WhisperASR()
    except Exception as e:
        print(f"Lỗi khởi tạo ASR: {e}")
        return None


def process_voice_input(audio_bytes):
    """Process recorded audio and convert to text using WhisperASR"""
    if not audio_bytes:
        return None

    asr = init_asr()
    if not asr:
        st.error("Không thể khởi tạo ASR. Vui lòng kiểm tra cài đặt.")
        return None

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        # Transcribe using WhisperASR (method is transcribe_audio, not transcribe)
        result = asr.transcribe_audio(temp_path, verbose=False)

        if result and result.get("full_text"):
            return result["full_text"].strip()
        else:
            return None
    except Exception as e:
        st.error(f"Lỗi nhận dạng giọng nói: {e}")
        return None
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except:
            pass


# =============================================================================
# Search Function
# =============================================================================

def semantic_search(query: str, top_k: int = 5):
    """Perform semantic search"""
    embedder = init_embedder()
    vector_db = init_vector_db()

    if not embedder or not vector_db:
        return []

    try:
        query_emb = embedder.encode_query(query)
        results = vector_db.hybrid_search(
            query=query,
            query_embedding=query_emb,
            alpha=0.7,
            top_k=top_k
        )
        return results
    except Exception as e:
        st.error(f"Lỗi tìm kiếm: {e}")
        return []


def get_answer(query: str, contexts: list) -> str:
    """Generate answer from contexts using LLM"""
    if not contexts:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

    # Try to use cached RAG system
    try:
        rag = init_rag()
        if rag:
            result = rag.query(query)
            if result.get("answer"):
                return result["answer"]
    except Exception as e:
        # LLM not available, fallback to showing contexts
        st.warning(f"LLM Error: {e}")
        import traceback
        print(f"RAG Error: {traceback.format_exc()}")

    # Fallback: Return relevant contexts
    context_text = "\n\n".join([
        f"**[{i+1}]** {ctx.get('text', '')[:300]}..."
        for i, ctx in enumerate(contexts[:3])
    ])

    return f"""**Thông tin tìm thấy:**

{context_text}

---
*Lưu ý: Để có câu trả lời tổng hợp, hệ thống cần kết nối với LLM (Ollama/Google/OpenAI).*
"""


# =============================================================================
# Main UI
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Hệ thống Tra cứu Thông tin</h1>
        <p>Đặt câu hỏi về quy định, học vụ, và các thông tin của nhà trường</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats bar
    stats = get_kb_stats()
    if stats and stats.total_documents > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Tài liệu", stats.total_documents)
        with col2:
            st.metric("📦 Đoạn văn bản", stats.total_chunks)
        with col3:
            st.metric("💾 Dung lượng", f"{stats.total_size_mb:.1f} MB")
    else:
        st.warning("⚠️ Chưa có dữ liệu trong hệ thống. Vui lòng liên hệ quản trị viên.")
        return

    st.divider()

    # Chat interface
    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(msg["content"])

                # Show sources
                # TODO: Implement timestamp display for audio/video sources (Chapter 4 design)
                # - Display start_time_formatted and end_time_formatted from source metadata
                # - Consider adding Source Player component for audio/video playback at specific timestamp
                if msg.get("sources"):
                    with st.expander("📚 Nguồn tham khảo", expanded=False):
                        for src in msg["sources"]:
                            similarity = src.get("similarity", 0)
                            text_preview = src.get("text", "")[:150]
                            # TODO: Add timestamp info: src.get("start_time_formatted"), src.get("end_time_formatted")
                            st.markdown(f"- **[{similarity:.0%}]** {text_preview}...")

                # TTS - simple: click button -> show audio directly
                if st.button("🔊 Nghe", key=f"tts_{i}"):
                    tts = init_tts()
                    if tts:
                        with st.spinner("Đang tạo audio..."):
                            try:
                                audio = tts.synthesize_sync(msg["content"][:500])
                                if audio:
                                    st.audio(audio, format="audio/mp3")
                            except Exception as e:
                                st.error(f"Lỗi TTS: {e}")

    # ==========================================================================
    # Input Section (Text + Voice)
    # ==========================================================================

    query = None
    query_source = None  # "text" or "voice"

    # Voice input section
    if VOICE_INPUT_AVAILABLE:
        # Voice hint
        st.markdown(
            '<div class="voice-hint">💡 <b>Mẹo:</b> Nhấn 🎤 và nói tiếng Việt rõ ràng. Dừng 2 giây để tự động gửi.</div>',
            unsafe_allow_html=True
        )

        col_input, col_voice = st.columns([9, 1])

        with col_voice:
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#1E88E5",
                icon_size="2x",
                pause_threshold=2.0,  # Auto-stop after 2s silence
            )

        # Process voice if recorded
        if audio_bytes:
            with st.spinner("🎤 Đang nhận dạng giọng nói..."):
                voice_text = process_voice_input(audio_bytes)

            if voice_text:
                st.markdown(
                    f'<div class="voice-transcript">🎤 <b>Bạn nói:</b> {voice_text}</div>',
                    unsafe_allow_html=True
                )
                query = voice_text
                query_source = "voice"
            else:
                st.warning("❌ Không nhận dạng được. Vui lòng nói rõ ràng hơn và thử lại.")

        with col_input:
            text_input = st.chat_input("Nhập câu hỏi hoặc nhấn 🎤 để nói...")
    else:
        # Fallback: text only if audio_recorder not available
        text_input = st.chat_input("Nhập câu hỏi của bạn (VD: Quy định đăng ký môn học?)")

    # Text input (if no voice query)
    if text_input and not query:
        query = text_input
        query_source = "text"

    # Process query (from either source)
    if query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user", avatar="👤"):
            if query_source == "voice":
                st.markdown(f"🎤 {query}")
            else:
                st.markdown(query)

        # Search and get answer
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Đang tìm kiếm..."):
                # Search
                results = semantic_search(query, top_k=5)

                # Get answer
                answer = get_answer(query, results)

                st.markdown(answer)

                # Show sources
                # TODO: Implement timestamp display for audio/video sources (Chapter 4 design)
                if results:
                    with st.expander("📚 Nguồn tham khảo", expanded=False):
                        for src in results[:3]:
                            similarity = src.get("similarity", 0)
                            text_preview = src.get("text", "")[:150]
                            # TODO: Add timestamp info for audio/video sources
                            st.markdown(f"- **[{similarity:.0%}]** {text_preview}...")

                # Auto-play TTS for voice queries (if enabled)
                if query_source == "voice" and st.session_state.auto_tts:
                    tts = init_tts()
                    if tts:
                        with st.spinner("🔊 Đang tạo audio..."):
                            try:
                                # Limit answer length for TTS
                                tts_text = answer[:800] if len(answer) > 800 else answer
                                audio_data = tts.synthesize_sync(tts_text)
                                if audio_data:
                                    st.audio(audio_data, format="audio/mp3", autoplay=True)
                            except Exception as e:
                                st.caption(f"⚠️ Không thể phát audio: {e}")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": results[:3] if results else []
        })

    # Sidebar
    with st.sidebar:
        # Voice settings
        if VOICE_INPUT_AVAILABLE:
            st.markdown("### 🎤 Cài đặt giọng nói")
            st.session_state.auto_tts = st.toggle(
                "🔊 Tự động đọc câu trả lời",
                value=st.session_state.auto_tts,
                help="Tự động phát audio khi hỏi bằng giọng nói"
            )
            st.divider()

        # Example questions
        st.markdown("### 💡 Câu hỏi mẫu")

        example_questions = [
            "Quy định đăng ký môn học?",
            "Điều kiện được thi cuối kỳ?",
            "Cách tính điểm trung bình?",
            "Quy định về tín chỉ tự chọn?",
            "Thời gian đăng ký môn học?",
        ]

        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

        st.divider()

        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("🎓 Hệ thống tra cứu thông tin")
        st.caption("© 2025 - Đồ án chuyên ngành")


if __name__ == "__main__":
    main()
