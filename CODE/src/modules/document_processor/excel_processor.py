"""
Excel Processor - Extract data from Excel files (.xlsx, .xls).
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .base import (
    BaseProcessor,
    ProcessedDocument,
    DocumentMetadata,
    TextChunk,
    Table,
)


class ExcelProcessor(BaseProcessor):
    """
    Processor cho Excel files (.xlsx, .xls).

    Features:
    - Extract data from all sheets
    - Convert to searchable text format
    - Preserve table structure
    - Extract metadata
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Excel Processor.

        Args:
            config: Configuration dict with options:
                - include_empty_cells: Include empty cells (default: False)
                - max_rows: Max rows per sheet (default: 10000)
                - sheet_names: List of sheet names to process (default: all)
        """
        super().__init__(config)
        self.include_empty = self.config.get("include_empty_cells", False)
        self.max_rows = self.config.get("max_rows", 10000)
        self.sheet_names = self.config.get("sheet_names", None)

    def supported_extensions(self) -> List[str]:
        return [".xlsx", ".xls"]

    def process(self, file_path: str) -> ProcessedDocument:
        """Process Excel file"""
        start_time = time.time()

        try:
            self.validate(file_path)
        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

        try:
            import pandas as pd
        except ImportError:
            return self._create_error_result(
                file_path,
                ImportError("pandas not installed. Run: pip install pandas openpyxl"),
                start_time
            )

        try:
            # Read Excel file
            xlsx = pd.ExcelFile(file_path)
            text_parts = []
            chunks = []
            tables = []
            char_offset = 0

            # Get sheets to process
            sheets_to_process = self.sheet_names or xlsx.sheet_names

            for sheet_idx, sheet_name in enumerate(sheets_to_process):
                if sheet_name not in xlsx.sheet_names:
                    continue

                try:
                    df = pd.read_excel(
                        xlsx,
                        sheet_name=sheet_name,
                        nrows=self.max_rows
                    )
                except Exception as e:
                    print(f"Error reading sheet {sheet_name}: {e}")
                    continue

                if df.empty:
                    continue

                # Clean data
                df = df.fillna("") if self.include_empty else df.dropna(how='all')

                # Convert to text
                sheet_header = f"### Sheet: {sheet_name}"
                text_parts.append(sheet_header)

                # Create readable text from dataframe
                sheet_text = self._dataframe_to_text(df)
                text_parts.append(sheet_text)

                # Create chunk for this sheet
                full_sheet_text = f"{sheet_header}\n{sheet_text}"
                chunks.append(TextChunk(
                    text=full_sheet_text,
                    start_char=char_offset,
                    end_char=char_offset + len(full_sheet_text),
                    chunk_index=sheet_idx,
                    metadata={"sheet_name": sheet_name}
                ))
                char_offset += len(full_sheet_text) + 2

                # Store as table
                tables.append(Table(
                    data=df.values.tolist(),
                    headers=df.columns.tolist(),
                    sheet_name=sheet_name,
                    table_index=sheet_idx
                ))

            # Metadata
            metadata = self.extract_metadata(file_path)
            metadata.sheet_count = len(xlsx.sheet_names)

            full_content = "\n\n".join(text_parts)

            return ProcessedDocument(
                content=full_content,
                chunks=chunks,
                metadata=metadata,
                source_file=file_path,
                file_type="xlsx",
                tables=tables,
                processed_at=datetime.now(),
                processing_time=time.time() - start_time,
                processor_version=self.VERSION,
                success=True
            )

        except Exception as e:
            return self._create_error_result(file_path, e, start_time)

    def _dataframe_to_text(self, df) -> str:
        """Convert DataFrame to readable text format"""
        lines = []

        # Add headers
        headers = df.columns.tolist()
        lines.append(" | ".join(str(h) for h in headers))
        lines.append("-" * 50)

        # Add rows
        for idx, row in df.iterrows():
            row_values = []
            for val in row.values:
                if val == "" or (hasattr(val, '__len__') and len(str(val)) == 0):
                    row_values.append("-")
                else:
                    row_values.append(str(val))
            lines.append(" | ".join(row_values))

        return "\n".join(lines)
