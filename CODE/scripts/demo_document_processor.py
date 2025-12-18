"""
Demo script for Document Processor Module
==========================================

This script demonstrates the capabilities of the document processor module:
- Processing different file types (PDF, Word, Excel, Image, Text)
- Batch processing
- Error handling

Usage:
    python scripts/demo_document_processor.py
    python scripts/demo_document_processor.py --file path/to/file.pdf
    python scripts/demo_document_processor.py --folder path/to/documents/
"""

import sys
import os
import argparse
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def print_separator(title: str = ""):
    """Print a separator line"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


def demo_unified_processor():
    """Demo UnifiedProcessor with sample files"""
    print_separator("UnifiedProcessor Demo")

    from src.modules.document_processor import UnifiedProcessor

    processor = UnifiedProcessor()

    # Show supported extensions
    print("\nSupported file types:")
    for category, extensions in processor.get_supported_types().items():
        print(f"  {category}: {', '.join(extensions)}")

    print("\nAll extensions:", processor.supported_extensions())


def demo_text_processor():
    """Demo TextProcessor with a sample text"""
    print_separator("TextProcessor Demo")

    from src.modules.document_processor import TextProcessor

    # Create a sample text file
    sample_text = """# Document-Based RAG Platform

## Introduction

This is a sample document for testing the Text Processor.
It supports multiple paragraphs and markdown formatting.

## Features

- Extract text from various file formats
- Support for Vietnamese language
- Automatic encoding detection

## Conclusion

The text processor is working correctly!
"""

    # Save sample file
    sample_path = "temp_sample.txt"
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    try:
        processor = TextProcessor()
        result = processor.process(sample_path)

        print(f"\nFile: {result.source_file}")
        print(f"File type: {result.file_type}")
        print(f"Success: {result.success}")
        print(f"Processing time: {result.processing_time:.3f}s")

        if result.metadata:
            print(f"\nMetadata:")
            print(f"  - Filename: {result.metadata.filename}")
            print(f"  - File size: {result.metadata.file_size} bytes")
            if result.metadata.extra:
                print(f"  - Line count: {result.metadata.extra.get('line_count', 'N/A')}")
                print(f"  - Word count: {result.metadata.extra.get('word_count', 'N/A')}")

        print(f"\nContent preview (first 200 chars):")
        print(result.content[:200] + "..." if len(result.content) > 200 else result.content)

        print(f"\nChunks: {len(result.chunks)}")
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"  Chunk {i+1}: {chunk.text[:50]}...")

    finally:
        # Cleanup
        if os.path.exists(sample_path):
            os.remove(sample_path)


def demo_pdf_processor(file_path: str = None):
    """Demo PDFProcessor"""
    print_separator("PDFProcessor Demo")

    from src.modules.document_processor import PDFProcessor

    if not file_path:
        print("No PDF file provided. Skipping PDF demo.")
        print("Usage: python demo_document_processor.py --file path/to/file.pdf")
        return

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        processor = PDFProcessor({
            "ocr_enabled": True,
            "extract_tables": True
        })

        print(f"\nProcessing: {file_path}")
        result = processor.process(file_path)

        print(f"\nResults:")
        print(f"  - Success: {result.success}")
        print(f"  - Processing time: {result.processing_time:.3f}s")

        if result.success:
            print(f"  - Pages: {result.metadata.page_count}")
            print(f"  - Content length: {len(result.content)} chars")
            print(f"  - Chunks: {len(result.chunks)}")
            print(f"  - Tables found: {len(result.tables)}")

            print(f"\nContent preview (first 300 chars):")
            print(result.content[:300] + "..." if len(result.content) > 300 else result.content)

            if result.tables:
                print(f"\nFirst table preview:")
                table = result.tables[0]
                for row in table.data[:3]:
                    print(f"  {row}")
        else:
            print(f"  - Error: {result.error_message}")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install PyMuPDF pdfplumber")


def demo_batch_processing(folder_path: str):
    """Demo batch processing of multiple files"""
    print_separator("Batch Processing Demo")

    from src.modules.document_processor import UnifiedProcessor

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    processor = UnifiedProcessor()

    # Find all supported files
    files = []
    for ext in processor.supported_extensions():
        files.extend(Path(folder_path).glob(f"*{ext}"))

    if not files:
        print(f"No supported files found in: {folder_path}")
        return

    print(f"\nFound {len(files)} files to process")

    # Progress callback
    def on_progress(current, total, file_path):
        print(f"  [{current}/{total}] {Path(file_path).name}")

    # Error callback
    def on_error(file_path, error):
        print(f"  ERROR in {Path(file_path).name}: {error}")

    # Process all files
    start_time = time.time()
    results = processor.process_batch(
        [str(f) for f in files],
        on_progress=on_progress,
        on_error=on_error
    )
    total_time = time.time() - start_time

    # Summary
    successful = sum(1 for r in results if r and r.success)
    failed = len(results) - successful

    print(f"\n--- Summary ---")
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per file: {total_time/len(files):.2f}s")


def demo_error_handling():
    """Demo error handling"""
    print_separator("Error Handling Demo")

    from src.modules.document_processor import UnifiedProcessor

    processor = UnifiedProcessor()

    # Test cases
    test_cases = [
        ("nonexistent.pdf", "File not found"),
        ("file.xyz", "Unsupported extension"),
    ]

    for file_path, expected_error in test_cases:
        print(f"\nTest: {file_path}")
        print(f"Expected: {expected_error}")

        try:
            result = processor.process(file_path)
            if result.success:
                print("Result: Unexpected success!")
            else:
                print(f"Result: Error - {result.error_message}")
        except Exception as e:
            print(f"Result: Exception - {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo Document Processor Module"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single file to process"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder for batch processing"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demos"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("   DOCUMENT PROCESSOR MODULE - DEMO")
    print("="*60)

    # Run demos
    demo_unified_processor()
    demo_text_processor()

    if args.file:
        demo_pdf_processor(args.file)

    if args.folder:
        demo_batch_processing(args.folder)

    demo_error_handling()

    print("\n" + "="*60)
    print("   Demo completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
