import json
import warnings  # Correct module for warnings, including PendingDeprecationWarning
import os
from pathlib import Path

from docx import Document
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from openai import OpenAI
from openai import OpenAI
from .llm_wrapper import LLMClientWrapper  # Relative import
from dotenv import load_dotenv
import tempfile

from ..config.splitter_config import SplitterConfig, LLM_Structured_Output_Type
from ..rlchunker.inference import RLChunker
from ..services.propositional_index import PropositionalIndexer

load_dotenv(override=True)
# Initialize OpenAI client
client =  OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from django.core.files.uploadedfile import UploadedFile

from enum import Enum

class SplitType(Enum):
    RECURSIVE = "RecursiveCharacterTextSplitter"
    CHARACTER = "CharacterTextSplitter"
    STANDARD = "standard"
    R_PRETRAINED_PROPOSITION = "RLBasedTextSplitterWithProposition"
    R_PRETRAINED = "RLBasedTextSplitter"

def extract_content_agnostic(file, filename=None):
    """
    Extract text content from a file.

    Supports:
    - PDF (.pdf)
    - Word (.docx)
    - Text (.txt)
    - Images (.png, .jpg, .jpeg, .tiff, .bmp)

    Parameters:
    - file: either a file path (str) or bytes (binary content)
    - filename: required if `file` is bytes, to determine extension
    """
    # Determine if file is path or binary
    if isinstance(file, str):
        filepath = file
        ext = os.path.splitext(filepath)[1].lower()
    elif isinstance(file, Path):
        filepath = str(file)
        ext = os.path.splitext(filepath)[1].lower()
    elif isinstance(file, bytes):
        if not filename:
            raise ValueError("filename must be provided if passing binary content")
        ext = os.path.splitext(filename)[1].lower()
        # Write bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file)
            filepath = tmp.name
    else:
        raise TypeError("file must be a string path or bytes")

    # At this point, `filepath` is a valid file path on disk
    # TODO: implement your extraction logic based on `ext` and `filepath`
    # Example:
    # if ext == ".pdf":
    #     content = extract_pdf(filepath)
    # elif ext == ".docx":
    #     content = extract_docx(filepath)
    # ...

    text = load_file_by_type(ext, filepath)

    # If we created a temporary file, optionally delete it
    if isinstance(file, UploadedFile) and not hasattr(file, 'temporary_file_path'):
        try:
            os.remove(filepath)
        except Exception:
            pass

    return text.strip()


def extract_content(file):
    """
    Extract text content from a file.
    Supports:
    - PDF (.pdf)
    - Word (.docx)
    - Text (.txt)
    - Images (.png, .jpg, .jpeg, .tiff, .bmp)

    file: either a file path (str) or a Django UploadedFile object (request.FILES['file'])
    """
    # Determine if input is file path or UploadedFile
    if isinstance(file, UploadedFile):
        filename = file.name
        ext = os.path.splitext(filename)[1].lower()

        # Check if file is already on disk
        if hasattr(file, 'temporary_file_path'):
            filepath = file.temporary_file_path()
        else:
            # Save in-memory file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in file.chunks():
                    tmp.write(chunk)
                filepath = tmp.name
    else:
        # It's a file path
        filepath = file
        ext = os.path.splitext(filepath)[1].lower()

    text = load_file_by_type(ext, filepath)

    # If we created a temporary file, optionally delete it
    if isinstance(file, UploadedFile) and not hasattr(file, 'temporary_file_path'):
        try:
            os.remove(filepath)
        except Exception:
            pass

    return text.strip()


def load_file_by_type(ext, filepath):
    text = ""
    if ext == ".pdf":
        reader = PdfReader(filepath)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])

    elif ext == ".docx":
        doc = Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])

    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".json":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            # Convert JSON to text (pretty print or flatten)
            text = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return text


def split_text_by_config(text, splitter_config:SplitterConfig=None, binary_data=None):
    """Split text into chunks of N words."""
    if splitter_config is None:
        splitter_config =  SplitterConfig(chunk_size= 200, chunk_overlap= 0)
        words = text.split()
        return [" ".join(words[i:i + splitter_config.chunk_size]) for i in
                range(0, len(words), splitter_config.chunk_size)]
    else:
        """Split text into chunks of N words."""
        if splitter_config.split_type == SplitType.STANDARD.value:
            words = text.split()
            return [" ".join(words[i:i + splitter_config.chunk_size]) for i in
                    range(0, len(words), splitter_config.chunk_size)]

        elif splitter_config.split_type == SplitType.RECURSIVE.value:
            """Split text into chunks of N characters."""
            text_splitter = RecursiveCharacterTextSplitter(
                separators=splitter_config.separators,
                chunk_size=splitter_config.chunk_size,
                chunk_overlap=splitter_config.chunk_overlap,
            )
            chunked_content = text_splitter.split_text(text)
            return chunked_content
        elif splitter_config.split_type == SplitType.CHARACTER.value:
            """Split text into chunks of N characters."""
            text_splitter = CharacterTextSplitter(
                separators=splitter_config.separators,
                chunk_size=splitter_config.chunk_size,
                chunk_overlap=splitter_config.chunk_overlap,
            )
            chunked_content = text_splitter.split_text(text)
            return chunked_content

        elif splitter_config.split_type == SplitType.R_PRETRAINED_PROPOSITION.value:
            indexer = PropositionalIndexer(model_name="gpt-4o-mini")

            # Index directly from file
            sentences = indexer.index_file_content(text, "propositional_index.txt")

            # ✅ Combine all sentences into one big text
            combined_text = " ".join(sentences)
            # Initialize chunker once
            chunker = RLChunker(device="cpu", embedding_dim=384)

            # Chunk a single text

            chunked_content = chunker.chunk_text(combined_text,min_len=splitter_config.min_rl_chunk_size,max_len=splitter_config.max_rl_chunk_size)

            return chunked_content
        elif splitter_config.split_type == SplitType.R_PRETRAINED.value:

            # Initialize chunker once
            chunker = RLChunker(device="cpu", embedding_dim=384)

            # Chunk a single text

            chunked_content = chunker.chunk_text(text,min_len=splitter_config.min_rl_chunk_size,max_len=splitter_config.max_rl_chunk_size)

            return chunked_content
        else:
            words = text.split()
            return [" ".join(words[i:i + splitter_config.chunk_size]) for i in
                    range(0, len(words), splitter_config.chunk_size)]


def process_with_llm(chunk,instructions):
    """
    Send a chunk to LLM and return structured JSON array.
    Expected format: [{"id": ..., "title": ..., "text": ...}, ...]
    """
    context = f"""
    Take the following text and split it into sections based on the most important category headings (ignore lower level headings).
    For each section, return a JSON object with - no extra words other than the json and remove ```json:
    - "id" (a UUID you generate),
    - "title" (the most important heading),
    - "text" (the remaining text under that heading).

    Text:
    {chunk}
    """
    instructions=instructions or "Exract sections"
    system_prompt="You are a helpful assistant that structures text into JSON sections."
    # Create an instance of your LLM wrapper
    llm = LLMClientWrapper(client, model="gpt-4o-mini", temperature=0, system_prompt=system_prompt)
    response=llm.chat(context,instructions)

    # Parse JSON safely
    try:
        structured_data = eval(response)
    except Exception:
        structured_data = []

    return structured_data


def process_large_text(text, instructions,splitter_config:SplitterConfig=None):
    """Main function: split -> send to LLM -> collect results."""
    chunks = split_text_by_config(text, splitter_config=splitter_config)
    all_results = []
    if splitter_config.enableLLMTouchUp:
        if splitter_config.llm_structured_output_type == LLM_Structured_Output_Type.STANDARD:
            warnings.warn("bypassing LLM touch up for standard structured output")
            return chunks
        elif splitter_config.llm_structured_output_type == LLM_Structured_Output_Type.STRUCTURED_WITH_VECTOR_DB_ID_GENERATED:
            for chunk in chunks:
                structured = process_with_llm(chunk,instructions)
                # Ensure UUIDs exist
                for obj in structured:
                    if "id" not in obj:
                        obj["id"] = str(uuid.uuid4())
                all_results.extend(structured)

            return all_results
    else:
        return chunks



def prepare_chunked_text(file_path,file_name,instructions,chunk_size=200,splitter_config:SplitterConfig=None):
    content =extract_content_agnostic(file_path,file_name)
    results=process_large_text(content,instructions, splitter_config=splitter_config)
    print (results)
    return results

#this function takes a django file and extracts filename and byte content
def extract_file_details(uploaded_file):
    # 1. Get the filename
    filename = uploaded_file.name

    # 2. Get the file content as bytes
    file_bytes = uploaded_file.read()  # reads entire file into memory

    # Now you can call your extract_content function
    return filename, file_bytes




















































