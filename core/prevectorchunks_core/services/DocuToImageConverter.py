import os
import tempfile
import base64
from pdf2image import convert_from_path
from docx2pdf import convert as docx_to_pdf
from openai import OpenAI
from PIL import Image
import io
# ---------------------------------------------------------
# 🧾 Class 1: Convert documents (PDF, DOCX, DOC) into images
# ---------------------------------------------------------
class DocuToImageConverter:
    """Converts a document (PDF, DOCX, DOC) into a list of PIL images."""

    def __init__(self):
        pass

    def _convert_doc_to_pdf(self, doc_path: str) -> str:
        """Converts a .docx or .doc file to PDF using docx2pdf."""
        temp_dir = tempfile.mkdtemp()
        output_pdf = os.path.join(temp_dir, "converted.pdf")
        docx_to_pdf(doc_path, output_pdf)
        return output_pdf

    def convert_to_images(self, file_path: str, dpi: int = 200, output_format: str = "PNG"):
        """
        Converts each page of a document into a list of PIL images.
        Supports .pdf, .doc, .docx, and image files (.jpg, .png, etc.)
        Ensures all outputs are in a consistent image format.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Convert Word → PDF first
        if ext in [".doc", ".docx"]:
            pdf_path = self._convert_doc_to_pdf(file_path)
            images = convert_from_path(pdf_path, dpi=dpi)

        # Convert PDF → list of images
        elif ext == ".pdf":
            images = convert_from_path(file_path, dpi=dpi)

        # Handle already an image
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            img = Image.open(file_path).convert("RGB")
            # Convert to consistent format (e.g., PNG or JPEG in memory)
            buffer = io.BytesIO()
            img.save(buffer, format=output_format)
            buffer.seek(0)
            converted_img = Image.open(buffer)
            images = [converted_img]

        else:
            raise ValueError("Unsupported file type. Use .pdf, .doc, .docx, or image files")

        return images
