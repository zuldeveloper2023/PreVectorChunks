import os
import tempfile
from PIL import Image
import io
from docx2pdf import convert as docx_to_pdf
import fitz



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

    def _convert_pdf_to_images(self, pdf_path: str, dpi: int = 200):
        """
        Converts each page of a PDF into images using PyMuPDF directly.
        """
        images = []

        try:
            pdf_document = fitz.open(pdf_path)  # Use `PyMuPDF` instead of fitz alias
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Render page to a pixmap with the specified DPI
                pixmap = page.get_pixmap(dpi=dpi)
                # Convert pixmap to an Image object using PIL
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                images.append(image)
            pdf_document.close()
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")

        return images

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
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)

        # Convert PDF → list of images
        elif ext == ".pdf":
            images = self._convert_pdf_to_images(file_path, dpi=dpi)

        # Handle already an image file
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            image = Image.open(file_path).convert("RGB")
            # Convert to consistent format (e.g., PNG or JPEG in memory)
            buffer = io.BytesIO()
            image.save(buffer, format=output_format)
            buffer.seek(0)
            converted_image = Image.open(buffer)
            images = [converted_image]

        else:
            raise ValueError("Unsupported file type. Use .pdf, .doc, .docx, or image files")

        return images
