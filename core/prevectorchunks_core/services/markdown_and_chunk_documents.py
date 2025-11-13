import os
import json
import tempfile
import uuid
from io import BytesIO
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

def get_file_extension(file_path,file_name):
    ext=''
    if file_name:
        ext = file_name[1]
    else:
        # Extract extension
        ext = os.path.splitext(file_path)[1].lower()
    return ext

# -----------------------------
# Abstract Strategy Interface
# -----------------------------
class BaseDocumentStrategy:
    """Defines a standard interface for all document processing strategies."""

    def process(self, file_path: str, input_bytes: bytes = None,ext:str=None):
        raise NotImplementedError("process() must be implemented by subclasses")

# -----------------------------
# PDF Strategy
# -----------------------------
class PDFStrategy(BaseDocumentStrategy):
    def process(self, file_path: str, input_bytes: bytes = None,ext:str=None):
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

        images = converter.convert_to_images(file_path,input_bytes,ext=ext)
        return images


# -----------------------------
# Word Strategy
# -----------------------------
class WordStrategy(BaseDocumentStrategy):
    def process(self, file_path: str, input_bytes: bytes = None,ext:str=None):
        file_name=''
        if file_path:
            file_name = Path(file_path)
            print(f"📝 Using WordStrategy for {file_path}")
        else:
            file_name_no_ext = os.path.splitext(input_bytes.name)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"{file_name}.pdf"

            converter = DocuToImageConverter()
            pdf_path = converter._convert_doc_to_pdf(file_path=file_path, input_bytes=input_bytes)
            images = converter.convert_to_images(pdf_path)


        return images


# -----------------------------
# Image Strategy
# -----------------------------
class ImageStrategy(BaseDocumentStrategy):
    def process(self, file_path: str, input_bytes: bytes = None,ext:str=None):
        print(f"🖼️ Using ImageStrategy for {file_path}")
        if file_path:
            # Path-based loading
            image = Image.open(file_path).convert("RGB")

        else:
            # Byte-based loading
            if input_bytes is None:
                raise ValueError("Either file_path or input_bytes must be provided")

            # If it's a Django UploadedFile → read() needed
            if hasattr(input_bytes, "read"):
                input_bytes.seek(0)
                image_bytes = input_bytes.read()

            # If it's already bytes
            elif isinstance(input_bytes, (bytes, bytearray)):
                image_bytes = input_bytes

            else:
                raise TypeError("input_bytes must be bytes or file-like object")

            image = Image.open(BytesIO(image_bytes)).convert("RGB")

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
    def get_strategy(cls, file_path: str,file_name:str=None) -> BaseDocumentStrategy:
        if file_name:
            ext=file_name[1]
        else:
            # Extract extension

            ext = os.path.splitext(file_path)[1].lower()

        return cls.strategies.get(ext, None)


# -----------------------------
# Main Orchestrator
# -----------------------------
class MarkdownAndChunkDocuments:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.extractor = DocuToMarkdownExtractor(api_key=self.api_key)

    def markdown_and_chunk_documents(self, file_path: str, input_bytes: bytes = None, include_image:bool=None,file_name:str=None):
        # Pick strategy
        strategy = StrategyFactory.get_strategy(file_path,file_name)
        if not strategy:
            raise ValueError(f"Unsupported file type: {file_path}")

        # Convert to images using correct strategy
        ext=get_file_extension(file_path,file_name)
        images = strategy.process(file_path, input_bytes,ext)

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
        adduuid(mapped_chunks)
        print("✅ Processing complete.")
        return mapped_chunks

    def markdown_and_chunk_documents_stream(
            self,
            file_path: str,
            input_bytes: bytes = None,
            include_image: bool = None,
            file_name: str = None,
    ):
        """Generator version of markdown_and_chunk_documents that yields progress JSON events"""

        def report(pct, msg=""):
            yield {"progress": int(pct), "status": msg}

        # 1️⃣ Pick strategy
        yield from report(5, "Selecting strategy...")
        strategy = StrategyFactory.get_strategy(file_path, file_name)
        if not strategy:
            raise ValueError(f"Unsupported file type: {file_path}")

        # 2️⃣ Convert to images
        ext = get_file_extension(file_path, file_name)
        yield from report(15, "Processing file into images...")
        images = strategy.process(file_path, input_bytes, ext)

        # 3️⃣ Extract Markdown
        yield from report(35, "Extracting markdown...")
        markdown_output, text_content = self.extractor.extract_markdown(images, include_image=include_image)
        binary_text_content = text_content.encode("utf-8")

        # 4️⃣ Chunking
        yield from report(55, "Chunking text...")
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

        chunked_text = chunk_documents(
            "", file_name="install_ins.txt", file_path=binary_text_content, splitter_config=splitter_config
        )
        flat_chunks = ["".join(inner) for inner in chunked_text]

        # 5️⃣ Map chunks (embedding)
        yield from report(60, f"Mapping {len(flat_chunks)} chunks...")
        total = len(flat_chunks)
        mapped_chunks = []
        for i, chunk in enumerate(flat_chunks, start=1):
            mapped = cm.map_chunks([chunk])
            mapped_chunks.extend(mapped)
            progress = 60 + (i / total) * 30
            yield from report(progress, f"Mapping chunk {i}/{total}")

        # 6️⃣ Merge unmapped markdown sections
        yield from report(95, "Merging markdown...")
        for md_item in markdown_output:
            if not any(md_item.get("markdown_text") == m.get("markdown_text") for m in mapped_chunks):
                md_item["chunked_text"] = md_item["markdown_text"]
                mapped_chunks.append(md_item)

        adduuid(mapped_chunks)
        yield from report(100, "✅ Processing complete.")

        # Final result
        yield {"progress": 100, "status": "done", "result": mapped_chunks}

def adduuid(mapped_chunks):
    # Assuming mapped_chunks is a list of dictionaries

    for chunk in mapped_chunks:
        chunk['id'] = str(uuid.uuid4())


# -----------------------------
# CLI Entry
# -----------------------------
if __name__ == "__main__":
    file_path = "421307-nz-au-top-loading-washer-guide-shorter.pdf"
    pipeline = MarkdownAndChunkDocuments()
    output = pipeline.markdown_and_chunk_documents(file_path)
    print(json.dumps(output, indent=2))
