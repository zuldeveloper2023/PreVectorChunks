import os
import json
import tempfile
from pathlib import Path

from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .DocuToImageConverter import DocuToImageConverter
from .DocuToMarkdownExtractor import DocuToMarkdownExtractor
from ..config.splitter_config import SplitterConfig
from .chunk_documents_crud_vdb import chunk_documents
from .chunk_to_all_content_mapper import ChunkMapper
from ..utils.file_loader import SplitType

load_dotenv(override=True)

# -----------------------------
# Abstract Strategy Interface
# -----------------------------
class BaseDocumentStrategy:
    """Defines a standard interface for all document processing strategies."""

    def process(self, file_path: str):
        raise NotImplementedError("process() must be implemented by subclasses")


# -----------------------------
# PDF Strategy
# -----------------------------
class PDFStrategy(BaseDocumentStrategy):
    def process(self, file_path: str):
        print(f"📄 Using PDFStrategy for {file_path}")
        converter = DocuToImageConverter()
        # Example: detect multi-column layout or extract embedded text first
        # import fitz
        # text_ratio = 0
        # with fitz.open(file_path) as doc:
        #     for page in doc:
        #         text = page.get_text("text")
        #         text_ratio += len(text) / (page.rect.width * page.rect.height)
        # if text_ratio > 0.0001:
        #     print("📚 PDF appears text-based – using hybrid extract + image backup")

        images = converter.convert_to_images(file_path)
        return images


# -----------------------------
# Word Strategy
# -----------------------------
class WordStrategy(BaseDocumentStrategy):
    def process(self, file_path: str):
        file_path = Path(file_path)

        print(f"📝 Using WordStrategy for {file_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"{file_path.stem}.pdf"

            converter = DocuToImageConverter()
            pdf_path = converter._convert_doc_to_pdf(file_path)
            images = converter.convert_to_images(pdf_path)


        return images


# -----------------------------
# Image Strategy
# -----------------------------
class ImageStrategy(BaseDocumentStrategy):
    def process(self, file_path: str):
        print(f"🖼️ Using ImageStrategy for {file_path}")
        image = Image.open(file_path).convert("RGB")
        return [image]


# -----------------------------
# Strategy Factory
# -----------------------------
class StrategyFactory:
    """Selects a document strategy based on file extension."""

    strategies = {
        ".pdf": PDFStrategy(),
        ".doc": WordStrategy(),
        ".docx": WordStrategy(),
        ".jpg": ImageStrategy(),
        ".jpeg": ImageStrategy(),
        ".png": ImageStrategy(),
        ".bmp": ImageStrategy(),
        ".tiff": ImageStrategy(),
    }

    @classmethod
    def get_strategy(cls, file_path: str) -> BaseDocumentStrategy:
        ext = os.path.splitext(file_path)[1].lower()
        return cls.strategies.get(ext, None)


# -----------------------------
# Main Orchestrator
# -----------------------------
class MarkdownAndChunkDocuments:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.extractor = DocuToMarkdownExtractor(api_key=self.api_key)

    def markdown_and_chunk_documents(self, file_path: str,include_image:bool):
        # Pick strategy
        strategy = StrategyFactory.get_strategy(file_path)
        if not strategy:
            raise ValueError(f"Unsupported file type: {file_path}")

        # Convert to images using correct strategy
        images = strategy.process(file_path)

        # Extract Markdown from images
        markdown_output, text_content = self.extractor.extract_markdown(images, include_image=include_image)
        binary_text_content = text_content.encode("utf-8")

        # Chunking and mapping
        chunk_client = OpenAI(api_key=self.api_key)
        cm = ChunkMapper(chunk_client, markdown_output, embedding_model="text-embedding-3-small")
        splitter_config = SplitterConfig(
            chunk_size=300,
            chunk_overlap=0,
            separators=["\n"],
            split_type=SplitType.R_PRETRAINED_PROPOSITION.value,
            min_rl_chunk_size=5,
            max_rl_chunk_size=50,
            enableLLMTouchUp=False,
        )

        chunked_text = chunk_documents("", file_name="install_ins.txt", file_path=binary_text_content,
                                       splitter_config=splitter_config)

        flat_chunks = [''.join(inner) for inner in chunked_text]
        mapped_chunks = cm.map_chunks(flat_chunks)

        # Merge unmapped markdown sections
        for md_item in markdown_output:
            if not any(md_item.get("markdown_text") == m.get("markdown_text") for m in mapped_chunks):
                md_item["chunked_text"] = md_item["markdown_text"]
                mapped_chunks.append(md_item)

        print("✅ Processing complete.")
        return mapped_chunks


# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    file_path = "421307-nz-au-top-loading-washer-guide-shorter.pdf"
    pipeline = MarkdownAndChunkDocuments()
    output = pipeline.markdown_and_chunk_documents(file_path)
    print(json.dumps(output, indent=2))
