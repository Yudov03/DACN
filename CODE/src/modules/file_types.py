"""
File Types Module
=================

Centralized definitions for all supported file types.
Single source of truth - import from here instead of defining locally.

Usage:
    from src.modules.file_types import (
        AUDIO_EXTENSIONS,
        VIDEO_EXTENSIONS,
        DOCUMENT_EXTENSIONS,
        ALL_SUPPORTED_EXTENSIONS,
        get_file_category,
    )
"""

# =============================================================================
# Audio Formats (Whisper ASR)
# =============================================================================
AUDIO_EXTENSIONS = frozenset({
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'
})

# =============================================================================
# Video Formats (FFmpeg + Whisper)
# =============================================================================
VIDEO_EXTENSIONS = frozenset({
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'
})

# =============================================================================
# Document Formats
# =============================================================================
PDF_EXTENSIONS = frozenset({'.pdf'})

WORD_EXTENSIONS = frozenset({'.docx', '.doc'})

EXCEL_EXTENSIONS = frozenset({'.xlsx', '.xls', '.csv', '.tsv'})

POWERPOINT_EXTENSIONS = frozenset({'.pptx', '.ppt'})

TEXT_EXTENSIONS = frozenset({
    '.txt', '.md', '.rst', '.rtf',
    '.html', '.htm', '.xml', '.json', '.yaml', '.yml',
    '.log', '.ini', '.cfg'
})

CODE_EXTENSIONS = frozenset({
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.java', '.kt',
    '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php', '.swift',
    '.cs', '.vb', '.sql',
    '.sh', '.bash', '.ps1',
    '.toml', '.r', '.R', '.scala'
})

IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'
})

EBOOK_EXTENSIONS = frozenset({'.epub'})

# =============================================================================
# Grouped Sets
# =============================================================================
DOCUMENT_EXTENSIONS = (
    PDF_EXTENSIONS |
    WORD_EXTENSIONS |
    EXCEL_EXTENSIONS |
    POWERPOINT_EXTENSIONS |
    TEXT_EXTENSIONS |
    CODE_EXTENSIONS |
    IMAGE_EXTENSIONS |
    EBOOK_EXTENSIONS
)

MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

ALL_SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | MEDIA_EXTENSIONS

# =============================================================================
# File Type to Folder Mapping (for Knowledge Base organization)
# =============================================================================
TYPE_TO_FOLDER = {
    # PDF
    'pdf': 'pdf',
    # Word
    'docx': 'word', 'doc': 'word',
    # Excel
    'xlsx': 'excel', 'xls': 'excel', 'csv': 'excel', 'tsv': 'excel',
    # PowerPoint
    'pptx': 'powerpoint', 'ppt': 'powerpoint',
    # Text/Data
    'txt': 'text', 'md': 'text', 'rst': 'text', 'rtf': 'text',
    'html': 'text', 'htm': 'text', 'xml': 'text',
    'json': 'text', 'yaml': 'text', 'yml': 'text',
    'log': 'text', 'ini': 'text', 'cfg': 'text',
    # Code
    'py': 'code', 'js': 'code', 'ts': 'code', 'jsx': 'code', 'tsx': 'code',
    'java': 'code', 'kt': 'code',
    'cpp': 'code', 'c': 'code', 'h': 'code', 'hpp': 'code',
    'go': 'code', 'rs': 'code', 'rb': 'code', 'php': 'code', 'swift': 'code',
    'cs': 'code', 'vb': 'code', 'sql': 'code',
    'sh': 'code', 'bash': 'code', 'ps1': 'code',
    'toml': 'code', 'r': 'code', 'R': 'code', 'scala': 'code',
    # Images (OCR)
    'png': 'image', 'jpg': 'image', 'jpeg': 'image',
    'bmp': 'image', 'tiff': 'image', 'tif': 'image', 'webp': 'image',
    # Ebook
    'epub': 'ebook',
    # Audio
    'mp3': 'audio', 'wav': 'audio', 'm4a': 'audio',
    'flac': 'audio', 'ogg': 'audio', 'wma': 'audio', 'aac': 'audio',
    # Video
    'mp4': 'video', 'avi': 'video', 'mkv': 'video', 'mov': 'video',
    'wmv': 'video', 'flv': 'video', 'webm': 'video', 'm4v': 'video',
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_file_category(extension: str) -> str:
    """
    Get the category of a file based on its extension.

    Args:
        extension: File extension with or without dot (e.g., '.pdf' or 'pdf')

    Returns:
        Category name: 'audio', 'video', 'document', or 'unknown'
    """
    ext = extension.lower()
    if not ext.startswith('.'):
        ext = '.' + ext

    if ext in AUDIO_EXTENSIONS:
        return 'audio'
    elif ext in VIDEO_EXTENSIONS:
        return 'video'
    elif ext in DOCUMENT_EXTENSIONS:
        return 'document'
    else:
        return 'unknown'


def get_folder_for_type(file_type: str) -> str:
    """
    Get the storage folder name for a file type.

    Args:
        file_type: File extension without dot (e.g., 'pdf', 'mp3')

    Returns:
        Folder name for storage
    """
    return TYPE_TO_FOLDER.get(file_type.lower(), 'other')


def is_media_file(extension: str) -> bool:
    """Check if file is audio or video."""
    ext = extension.lower()
    if not ext.startswith('.'):
        ext = '.' + ext
    return ext in MEDIA_EXTENSIONS


def is_audio_file(extension: str) -> bool:
    """Check if file is audio."""
    ext = extension.lower()
    if not ext.startswith('.'):
        ext = '.' + ext
    return ext in AUDIO_EXTENSIONS


def is_video_file(extension: str) -> bool:
    """Check if file is video."""
    ext = extension.lower()
    if not ext.startswith('.'):
        ext = '.' + ext
    return ext in VIDEO_EXTENSIONS


# Extension sets without dots (for compatibility)
AUDIO_TYPES = frozenset(ext.lstrip('.') for ext in AUDIO_EXTENSIONS)
VIDEO_TYPES = frozenset(ext.lstrip('.') for ext in VIDEO_EXTENSIONS)
MEDIA_TYPES = AUDIO_TYPES | VIDEO_TYPES
