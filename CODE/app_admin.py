"""
Admin Portal - Hệ thống quản lý Knowledge Base
==============================================

Giao diện dành cho QUẢN TRỊ VIÊN (nhà trường) để:
- Upload tài liệu (PDF, Word, Audio, Video...)
- Quản lý Knowledge Base
- Import từ data/resource/
- Xem thống kê hệ thống

Run with: streamlit run app_admin.py --server.port 8502
"""

import streamlit as st
import sys
import io
from pathlib import Path
from datetime import datetime
import shutil

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Admin - Quản lý Knowledge Base",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        background: linear-gradient(135deg, #FF6B35, #F7931E);
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

    /* Stats cards */
    .stat-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border-left: 4px solid #FF6B35;
    }

    /* Success/Error messages */
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .error-msg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    /* Document list */
    .doc-item {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1E88E5;
    }

    /* Hide unnecessary elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
RESOURCE_DIR = PROJECT_ROOT / "data" / "resource"
RESOURCE_DOCS_DIR = RESOURCE_DIR / "documents"
RESOURCE_AUDIO_DIR = RESOURCE_DIR / "audio"
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

# Supported extensions
DOC_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.rst', '.rtf',
                  '.xlsx', '.xls', '.csv', '.tsv', '.pptx', '.ppt',
                  '.html', '.htm', '.xml', '.json', '.yaml', '.yml',
                  '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs',
                  '.epub', '.png', '.jpg', '.jpeg'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.flv', '.m4v'}

ALL_EXTENSIONS = DOC_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# =============================================================================
# Helper Functions
# =============================================================================

def ensure_directories():
    """Ensure all required directories exist"""
    RESOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)


def get_kb_stats():
    """Get Knowledge Base statistics"""
    try:
        from src.modules import KnowledgeBase
        kb = KnowledgeBase(base_dir=str(KB_DIR))
        return kb.get_stats()
    except Exception as e:
        return None


def get_kb():
    """Get KnowledgeBase instance"""
    try:
        from src.modules import KnowledgeBase
        return KnowledgeBase(base_dir=str(KB_DIR))
    except Exception as e:
        st.error(f"Loi khoi tao Knowledge Base: {e}")
        return None


def scan_resource_files():
    """Scan resource folder for supported files"""
    files = []
    for file_path in RESOURCE_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ALL_EXTENSIONS:
            files.append(file_path)
    return sorted(files)


def get_file_category(file_path: Path) -> str:
    """Get category of file based on extension"""
    ext = file_path.suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    else:
        return "document"


def format_size(size_bytes: int) -> str:
    """Format file size"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


# =============================================================================
# Main UI Components
# =============================================================================

def render_header():
    """Render header"""
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Admin Portal - Quan ly Knowledge Base</h1>
        <p>Upload tai lieu, quan ly du lieu va cau hinh he thong</p>
    </div>
    """, unsafe_allow_html=True)


def render_stats():
    """Render statistics"""
    stats = get_kb_stats()
    resource_files = scan_resource_files()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📁 Resource Files",
            len(resource_files),
            help="Files trong data/resource/"
        )

    with col2:
        if stats:
            st.metric(
                "📄 KB Documents",
                stats.total_documents,
                help="Documents trong Knowledge Base"
            )
        else:
            st.metric("📄 KB Documents", 0)

    with col3:
        if stats:
            st.metric(
                "📦 Chunks",
                stats.total_chunks,
                help="So luong text chunks"
            )
        else:
            st.metric("📦 Chunks", 0)

    with col4:
        if stats:
            st.metric(
                "💾 KB Size",
                f"{stats.total_size_mb:.1f} MB"
            )
        else:
            st.metric("💾 KB Size", "0 MB")


def render_upload_section():
    """Render file upload section"""
    st.subheader("📤 Upload tai lieu")

    tab1, tab2 = st.tabs(["Upload truc tiep", "Import tu Resource"])

    with tab1:
        render_direct_upload()

    with tab2:
        render_import_from_resource()


def render_direct_upload():
    """Direct file upload"""
    uploaded_files = st.file_uploader(
        "Chon files de upload",
        type=[ext.replace('.', '') for ext in ALL_EXTENSIONS],
        accept_multiple_files=True,
        help="Ho tro: PDF, Word, Excel, Audio, Video..."
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} files duoc chon:**")

        for f in uploaded_files:
            ext = Path(f.name).suffix.lower()
            category = get_file_category(Path(f.name))
            icon = "🎵" if category == "audio" else "🎬" if category == "video" else "📄"
            st.write(f"  {icon} {f.name} ({format_size(f.size)})")

        col1, col2 = st.columns(2)

        with col1:
            upload_to_resource = st.checkbox(
                "Luu vao Resource folder",
                value=True,
                help="Luu files vao data/resource/"
            )

        with col2:
            import_to_kb = st.checkbox(
                "Import vao Knowledge Base",
                value=True,
                help="Import truc tiep vao KB"
            )

        if st.button("⬆️ Upload", type="primary", use_container_width=True):
            ensure_directories()

            success_count = 0
            error_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Determine target directory
                    category = get_file_category(Path(uploaded_file.name))
                    if category in ["audio", "video"]:
                        target_dir = RESOURCE_AUDIO_DIR
                    else:
                        target_dir = RESOURCE_DOCS_DIR

                    # Save file
                    file_path = target_dir / uploaded_file.name

                    if upload_to_resource:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        status_text.text(f"Saved: {uploaded_file.name}")

                    # Import to KB
                    if import_to_kb:
                        kb = get_kb()
                        if kb:
                            tags = [category, Path(uploaded_file.name).suffix.lower().replace('.', '')]
                            kb.add_document(str(file_path), tags=tags)

                    success_count += 1

                except Exception as e:
                    st.error(f"Loi upload {uploaded_file.name}: {e}")
                    error_count += 1

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.empty()
            progress_bar.empty()

            if success_count > 0:
                st.success(f"✅ Upload thanh cong {success_count} files!")
            if error_count > 0:
                st.error(f"❌ {error_count} files that bai")

            st.rerun()


def render_import_from_resource():
    """Import files from resource folder"""
    st.write("**Import tat ca files tu `data/resource/` vao Knowledge Base**")

    resource_files = scan_resource_files()

    if not resource_files:
        st.info("📂 Khong co files trong data/resource/. Upload files hoac copy truc tiep vao folder.")
        return

    # Categorize files
    docs = [f for f in resource_files if get_file_category(f) == "document"]
    audios = [f for f in resource_files if get_file_category(f) == "audio"]
    videos = [f for f in resource_files if get_file_category(f) == "video"]

    st.write(f"""
    **Tim thay {len(resource_files)} files:**
    - 📄 Documents: {len(docs)}
    - 🎵 Audio: {len(audios)}
    - 🎬 Video: {len(videos)}
    """)

    with st.expander("📋 Xem danh sach files", expanded=False):
        for f in resource_files:
            rel_path = f.relative_to(RESOURCE_DIR)
            size = format_size(f.stat().st_size)
            icon = "🎵" if get_file_category(f) == "audio" else "🎬" if get_file_category(f) == "video" else "📄"
            st.write(f"  {icon} `{rel_path}` ({size})")

    col1, col2 = st.columns(2)

    with col1:
        clear_kb_first = st.checkbox(
            "Xoa KB truoc khi import",
            value=False,
            help="Xoa toan bo Knowledge Base truoc khi import"
        )

    if st.button("📥 Import tat ca vao Knowledge Base", type="primary", use_container_width=True):
        kb = get_kb()
        if not kb:
            return

        # Clear if requested
        if clear_kb_first:
            with st.spinner("Dang xoa Knowledge Base cu..."):
                existing_docs = kb.list_documents()
                for doc in existing_docs:
                    try:
                        kb.remove_document(doc.get('id', ''))
                    except:
                        pass
                st.info(f"Da xoa {len(existing_docs)} documents cu")

        # Import files
        success_count = 0
        error_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file_path in enumerate(resource_files):
            try:
                rel_path = file_path.relative_to(RESOURCE_DIR)
                status_text.text(f"Importing: {rel_path}")

                # Determine tags
                category = get_file_category(file_path)
                tags = [category, file_path.suffix.lower().replace('.', '')]

                # Add to KB
                doc_id = kb.add_document(str(file_path), tags=tags)
                success_count += 1

            except Exception as e:
                error_count += 1
                st.warning(f"Skip {file_path.name}: {e}")

            progress_bar.progress((i + 1) / len(resource_files))

        status_text.empty()
        progress_bar.empty()

        if success_count > 0:
            st.success(f"✅ Import thanh cong {success_count} files!")
        if error_count > 0:
            st.warning(f"⚠️ {error_count} files bi skip")

        st.rerun()


def render_kb_management():
    """Render Knowledge Base management section"""
    st.subheader("📚 Quan ly Knowledge Base")

    kb = get_kb()
    if not kb:
        return

    docs = kb.list_documents()

    if not docs:
        st.info("📭 Knowledge Base trong. Hay upload tai lieu!")
        return

    st.write(f"**{len(docs)} documents trong Knowledge Base:**")

    # Search/filter
    search_query = st.text_input("🔍 Tim kiem document...", placeholder="Nhap ten file hoac tag...")

    # Filter docs
    if search_query:
        filtered_docs = [
            d for d in docs
            if search_query.lower() in d.get('filename', '').lower()
            or search_query.lower() in ' '.join(d.get('tags', [])).lower()
        ]
    else:
        filtered_docs = docs

    # Display docs
    for doc in filtered_docs:
        with st.container():
            col1, col2, col3 = st.columns([4, 2, 1])

            with col1:
                filename = doc.get('filename', 'Unknown')
                doc_type = doc.get('type', 'unknown')
                icon = "🎵" if doc_type == "audio" else "🎬" if doc_type == "video" else "📄"
                st.write(f"{icon} **{filename}**")

                tags = doc.get('tags', [])
                if tags:
                    st.caption(f"Tags: {', '.join(tags)}")

            with col2:
                chunks = doc.get('chunk_count', 0)
                added = doc.get('added_at', '')
                if added:
                    try:
                        dt = datetime.fromisoformat(added)
                        added_str = dt.strftime("%d/%m/%Y")
                    except:
                        added_str = added[:10]
                else:
                    added_str = "N/A"

                st.caption(f"📦 {chunks} chunks | 📅 {added_str}")

            with col3:
                if st.button("🗑️", key=f"del_{doc.get('id', '')}", help="Xoa document"):
                    try:
                        kb.remove_document(doc.get('id', ''))
                        st.success(f"Da xoa: {filename}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Loi: {e}")

            st.divider()

    # Bulk actions
    st.write("---")
    st.write("**Thao tac hang loat:**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Re-index tat ca", use_container_width=True, help="Index lai toan bo documents"):
            st.info("Chuc nang dang phat trien...")

    with col2:
        if st.button("🗑️ Xoa toan bo KB", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_delete_all"):
                # Actually delete
                for doc in docs:
                    try:
                        kb.remove_document(doc.get('id', ''))
                    except:
                        pass
                st.success("Da xoa toan bo Knowledge Base!")
                st.session_state.confirm_delete_all = False
                st.rerun()
            else:
                st.session_state.confirm_delete_all = True
                st.warning("⚠️ Click lan nua de xac nhan XOA TAT CA!")


def render_system_info():
    """Render system information"""
    st.subheader("ℹ️ Thong tin he thong")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Cau hinh:**")

        # Check providers
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)

        emb_provider = os.getenv("EMBEDDING_PROVIDER", "local")
        llm_provider = os.getenv("LLM_PROVIDER", "ollama")

        st.write(f"- Embedding Provider: `{emb_provider}`")
        st.write(f"- LLM Provider: `{llm_provider}`")

        # Check Qdrant
        try:
            from src.modules import VectorDatabase
            vdb = VectorDatabase(collection_name="test_connection", embedding_dimension=768)
            st.write("- Qdrant: ✅ Connected")
        except:
            st.write("- Qdrant: ⚠️ Offline (using in-memory)")

    with col2:
        st.write("**Duong dan:**")
        st.code(f"""
Resource:     {RESOURCE_DIR}
Knowledge:    {KB_DIR}
        """)

        st.write("**Huong dan:**")
        st.write("""
        1. Upload files qua web hoac copy vao `data/resource/`
        2. Click "Import vao Knowledge Base"
        3. Sinh vien truy cap `app.py` de tra cuu
        """)


def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.markdown("### 🔧 Admin Menu")

        st.divider()

        st.write("**Quick Actions:**")

        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

        if st.button("📂 Mo Resource folder", use_container_width=True):
            import subprocess
            try:
                subprocess.Popen(f'explorer "{RESOURCE_DIR}"', shell=True)
            except:
                st.info(f"Path: {RESOURCE_DIR}")

        st.divider()

        st.write("**Links:**")
        st.page_link("app.py", label="👤 Student Portal", icon="🎓")

        st.divider()

        st.caption("⚙️ Admin Portal")
        st.caption("© 2025 - Do an chuyen nganh")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_directories()

    render_header()
    render_stats()

    st.divider()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📚 Quan ly KB", "ℹ️ He thong"])

    with tab1:
        render_upload_section()

    with tab2:
        render_kb_management()

    with tab3:
        render_system_info()

    # Sidebar
    render_sidebar()


if __name__ == "__main__":
    main()
