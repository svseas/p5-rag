"""
Marker-based PDF parser for complex document layouts.

Marker is a vision-based PDF parser that uses LLMs to understand document structure,
making it superior for tables, equations, and complex layouts compared to traditional OCR.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser as MarkerConfigParser

logger = logging.getLogger(__name__)


class MarkerParser:
    """
    Marker-based PDF parser that uses vision models for structure-aware extraction.

    Benefits over traditional OCR:
    - Preserves table structure properly (row/column alignment)
    - Better handling of complex layouts (multi-column, mixed content)
    - Vision model understands spatial relationships
    - Outputs clean Markdown with preserved formatting
    """

    def __init__(
        self,
        output_format: str = "markdown",
        llm_model: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        **marker_config: Dict[str, Any]
    ):
        """
        Initialize Marker parser.

        Args:
            output_format: Output format - "markdown", "json", or "html"
            llm_model: LLM model name (e.g., "qwen2.5-vl:7b")
            llm_api_base: API base URL for hosted LLM (e.g., "http://localhost:11434/v1")
            llm_api_key: API key for LLM service (optional)
            **marker_config: Additional Marker configuration options
        """
        self.output_format = output_format
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.marker_config = marker_config

        logger.info(
            f"Initialized MarkerParser with output_format={output_format}, "
            f"llm_model={llm_model}"
        )

    def parse_pdf(self, file_content: bytes, filename: str) -> str:
        """
        Parse PDF using Marker vision-based extraction.

        Args:
            file_content: PDF file bytes
            filename: Original filename (for metadata)

        Returns:
            Extracted text in specified format (default: Markdown)

        Raises:
            Exception: If parsing fails
        """
        try:
            # Marker requires a file path, so write to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            try:
                # Build Marker configuration
                config = {
                    "output_format": self.output_format,
                    **self.marker_config
                }

                # Add LLM service configuration if provided
                if self.llm_model:
                    config["llm_model"] = self.llm_model
                if self.llm_api_base:
                    config["llm_api_base"] = self.llm_api_base
                if self.llm_api_key:
                    config["llm_api_key"] = self.llm_api_key

                # Create Marker config parser
                config_parser = MarkerConfigParser(config)

                # Initialize Marker converter
                converter = PdfConverter(
                    config=config_parser.generate_config_dict(),
                    artifact_dict=create_model_dict(),
                    processor_list=config_parser.get_processors(),
                    renderer=config_parser.get_renderer(),
                    llm_service=config_parser.get_llm_service()
                )

                # Convert PDF
                logger.info(f"Processing {filename} with Marker (format={self.output_format})")
                rendered = converter(temp_path)

                # Extract text based on format
                if self.output_format == "markdown":
                    text = rendered.markdown if hasattr(rendered, 'markdown') else str(rendered)
                elif self.output_format == "json":
                    text = rendered.json() if hasattr(rendered, 'json') else str(rendered)
                else:
                    text = str(rendered)

                logger.info(f"Marker successfully parsed {filename} ({len(text)} characters)")
                return text

            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Marker parsing failed for {filename}: {e}", exc_info=True)
            raise

    def parse(self, file_content: bytes, filename: str, content_type: str) -> str:
        """
        Parse document (currently only supports PDF).

        Args:
            file_content: File bytes
            filename: Original filename
            content_type: MIME type

        Returns:
            Extracted text

        Raises:
            ValueError: If content type is not PDF
            Exception: If parsing fails
        """
        if not filename.lower().endswith('.pdf') and content_type != 'application/pdf':
            raise ValueError(f"Marker parser only supports PDF files, got {content_type}")

        return self.parse_pdf(file_content, filename)
