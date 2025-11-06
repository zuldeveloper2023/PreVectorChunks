import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pypandoc
from PIL import Image
import io
from docx2pdf import convert as docx_to_pdf
import fitz
from docx2pdf import convert as docx2pdf_convert
try:
    pypandoc.get_pandoc_path()
except OSError:
    print("Pandoc not found — downloading it temporarily...")
    pypandoc.download_pandoc()

class DocuToImageConverter:
    """Converts a document (PDF, DOCX, DOC) into a list of PIL images."""

    def __init__(self):
        pass

    def _convert_doc_to_pdf(self, input_path: str) -> str:
        import os, tempfile, shutil, subprocess
        from pathlib import Path

        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)

        output_dir = tempfile.mkdtemp()
        output_pdf = os.path.join(output_dir, Path(input_path).stem + ".pdf")

        # 1️⃣ Try Microsoft Word COM automation (Windows only)
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(str(Path(input_path).resolve()))
            doc.SaveAs(str(Path(output_pdf).resolve()), FileFormat=17)  # 17 = wdFormatPDF
            doc.Close()
            word.Quit()
            print("✅ Word COM conversion successful:", output_pdf)
            return output_pdf
        except Exception as e:
            print("⚠️ Word COM conversion failed:", e)

        # 2️⃣ Fallback: LibreOffice (cross-platform, preserves layout)
        try:
            # Requires LibreOffice installed and in PATH
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", output_dir, input_path],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print("✅ LibreOffice conversion successful:", output_pdf)
            return output_pdf
        except Exception as e:
            print("⚠️ LibreOffice conversion failed:", e)

        # 3️⃣ Fallback: Pandoc (simpler, loses layout)
        try:
            import pypandoc
            def which(cmd):
                return shutil.which(cmd) is not None

            pdf_engine = "pdflatex" if which("pdflatex") else "wkhtmltopdf"
            pypandoc.convert_file(
                input_path, "pdf", outputfile=output_pdf,
                extra_args=["--standalone", f"--pdf-engine={pdf_engine}"]
            )
            print("✅ Pandoc conversion successful:", output_pdf)
            return output_pdf
        except Exception as e:
            print("⚠️ Pandoc conversion failed:", e)

        # 4️⃣ Last resort: ReportLab basic text (no formatting)
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from docx import Document

        doc = Document(input_path)
        c = canvas.Canvas(output_pdf, pagesize=A4)
        width, height = A4
        y = height - 50
        for p in doc.paragraphs:
            c.drawString(50, y, p.text[:1000])
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50
        c.save()
        print("⚠️ Fallback to plain ReportLab text output:", output_pdf)
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
