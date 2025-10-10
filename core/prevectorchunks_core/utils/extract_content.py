import json
import os
import tempfile
from pathlib import Path

from PIL import Image
from PyPDF2 import PdfReader
from django.core.files.uploadedfile import UploadedFile
from docx import Document
from pytesseract import pytesseract


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

