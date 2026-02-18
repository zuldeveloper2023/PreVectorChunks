import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from PIL import Image
import io
import fitz
from docx2pdf import convert as docx2pdf_convert
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import pypandoc

# Ensure pandoc is available
try:
    pypandoc.get_pandoc_path()
except OSError:
    pypandoc.download_pandoc()

class DocuToImageConverter:
    """Converts a document (PDF, DOCX, DOC, image bytes) into a list of PIL images."""

    def __init__(self):
        pass

    def _write_temp_file(self, input_bytes: bytes, suffix: str):
        """Write bytes to a temporary file and return path."""
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(input_bytes.read())
        return tmp_path

    def _convert_doc_to_pdf(self, file_path: str = None, input_bytes=None) -> str:
        """
        Convert DOC/DOCX to PDF. Supports:
        - file_path (string)
        - input_bytes (bytes, InMemoryUploadedFile, or file-like)
        """

        # ✅ If bytes are provided, write them to a temporary .docx
        if input_bytes is not None:
            # Get filename or fallback
            original_name = getattr(input_bytes, "name", "uploaded.docx")
            ext = os.path.splitext(original_name)[1] or ".docx"

            # Create a temporary file path
            temp_input_path = tempfile.mktemp(suffix=ext)

            # Read bytes safely
            if hasattr(input_bytes, "read"):  # Django UploadedFile
                input_bytes.seek(0)
                content = input_bytes.read()
            else:  # already bytes
                content = input_bytes

            # Write bytes to temp file
            with open(temp_input_path, "wb") as f:
                f.write(content)

            input_path = temp_input_path

        # ✅ If file_path is provided, use it directly
        elif file_path:
            input_path = file_path

        else:
            raise ValueError("Must supply either file_path or input_bytes")

        # ✅ Must exist at this point
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)

        # ✅ Prepare output PDF path
        output_dir = tempfile.mkdtemp()
        output_pdf = os.path.join(output_dir, Path(input_path).stem + ".pdf")

        # 1️⃣ Try Microsoft Word COM automation (Windows)
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(str(Path(input_path).resolve()))
            doc.SaveAs(str(Path(output_pdf).resolve()), FileFormat=17)
            doc.Close()
            word.Quit()
            return output_pdf
        except Exception:
            pass

        # 2️⃣ Try LibreOffice
        try:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", output_dir, input_path],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return output_pdf
        except Exception:
            pass

        # 3️⃣ Pandoc fallback
        try:
            pdf_engine = "pdflatex" if shutil.which("pdflatex") else "wkhtmltopdf"
            pypandoc.convert_file(
                input_path, "pdf",
                outputfile=output_pdf,
                extra_args=["--standalone", f"--pdf-engine={pdf_engine}"]
            )
            return output_pdf
        except Exception:
            pass

        # 4️⃣ Final fallback: Render plain text using ReportLab
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
        return output_pdf

    def _convert_pdf_to_images(self, pdf_path: str, dpi: int = 200):
        images = []
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            images.append(image)
        pdf_document.close()
        return images

    def _convert_txt_to_pdf(self, file_path: str = None, input_bytes=None) -> str:
        """
        Convert a .txt file or text bytes to PDF using ReportLab.
        """
        # Read text
        if input_bytes is not None:
            if hasattr(input_bytes, "read"):
                input_bytes.seek(0)
                text = input_bytes.read().decode("utf-8")
            else:
                text = input_bytes.decode("utf-8")
        elif file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError("Must supply either file_path or input_bytes")

        # Prepare output PDF
        output_dir = tempfile.mkdtemp()
        output_pdf = os.path.join(output_dir, (Path(file_path).stem if file_path else "temp") + ".pdf")

        # Write text to PDF
        c = canvas.Canvas(output_pdf, pagesize=A4)
        width, height = A4
        y = height - 50

        for line in text.splitlines():
            c.drawString(50, y, line[:1000])  # truncate very long lines
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50

        c.save()
        return output_pdf

    def _convert_csv_to_pdf(self, file_path: str = None, input_bytes=None) -> str:
        """
        Convert a CSV file or CSV bytes to PDF using ReportLab tables.
        """
        # Read CSV data
        rows = []
        if input_bytes is not None:
            if hasattr(input_bytes, "read"):
                input_bytes.seek(0)
                reader = csv.reader(io.StringIO(input_bytes.read().decode("utf-8")))
            else:
                reader = csv.reader(io.StringIO(input_bytes.decode("utf-8")))
            rows = list(reader)
        elif file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        else:
            raise ValueError("Must supply either file_path or input_bytes")

        if not rows:
            raise ValueError("CSV is empty")

        # Prepare output PDF
        output_dir = tempfile.mkdtemp()
        output_pdf = os.path.join(output_dir, (Path(file_path).stem if file_path else "temp") + ".pdf")

        # Create a table PDF
        doc = SimpleDocTemplate(output_pdf, pagesize=A4)
        table = Table(rows, repeatRows=1)

        # Style table
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ])
        table.setStyle(style)

        elements = [table]
        doc.build(elements)

        return output_pdf
    def convert_to_images(self, file_path: str = None, input_bytes: bytes = None, dpi: int = 200, output_format: str = "PNG",ext:str=None):
        """
        Convert a file path or binary content to PIL images.
        Supports PDF, DOC, DOCX, and image files.
        """
        if not file_path and not input_bytes:
            raise ValueError("Provide either file_path or input_bytes.")

        # Determine extension
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            print('work')
        elif input_bytes:
            # Attempt to infer from first few bytes (simple)
            # if input_bytes[:4] == b"%PDF":
            #     ext = ".pdf"
            # elif input_bytes[:2] == b"PK":
            #     ext = ".docx"
            # else:
            #     ext = ".img"  # Treat as generic image

            # Write to temp file if doc/pdf
            if ext in [".pdf", ".doc", ".docx"]:
                file_path = self._write_temp_file(input_bytes, suffix=ext)

        # Word → PDF
        if ext in [".doc", ".docx"]:
            pdf_path = self._convert_doc_to_pdf(file_path, input_bytes)
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)

        # TXT → PDF
        elif ext == ".txt":
            pdf_path = self._convert_txt_to_pdf(file_path, input_bytes)
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)
        # CSV → PDF
        elif ext == ".csv":
            pdf_path = self._convert_csv_to_pdf(file_path, input_bytes)
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)

        # PowerPoint → PDF
        elif ext in [".ppt", ".pptx"]:
            pdf_path = self._convert_ppt_to_pdf(file_path, input_bytes)
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)

        # Excel → PDF
        elif ext in [".xls", ".xlsx"]:
            pdf_path = self._convert_excel_to_pdf(file_path, input_bytes)
            images = self._convert_pdf_to_images(pdf_path, dpi=dpi)


        # PDF → images
        elif ext == ".pdf":
            images = self._convert_pdf_to_images(file_path, dpi=dpi)

        # Image
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".img"]:
            image = Image.open(io.BytesIO(input_bytes) if input_bytes else file_path).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format=output_format)
            buffer.seek(0)
            images = [Image.open(buffer)]

        else:
            raise ValueError("Unsupported file type.")

        return images

    def _convert_ppt_to_pdf(self, file_path: str = None, input_bytes=None) -> str:
        """
        Convert PPT/PPTX to PDF using:
        1. PowerPoint COM (Windows)
        2. LibreOffice
        """

        # write bytes if needed
        if input_bytes is not None:
            original_name = getattr(input_bytes, "name", "uploaded.pptx")
            ext = os.path.splitext(original_name)[1] or ".pptx"
            temp_input_path = tempfile.mktemp(suffix=ext)

            if hasattr(input_bytes, "read"):
                input_bytes.seek(0)
                content = input_bytes.read()
            else:
                content = input_bytes

            with open(temp_input_path, "wb") as f:
                f.write(content)

            input_path = temp_input_path

        elif file_path:
            input_path = file_path

        else:
            raise ValueError("Must supply either file_path or input_bytes")

        output_dir = tempfile.mkdtemp()
        output_pdf = os.path.join(output_dir, Path(input_path).stem + ".pdf")

        # 1️⃣ Try PowerPoint COM on Windows
        try:
            import win32com.client
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            powerpoint.Visible = 1

            deck = powerpoint.Presentations.Open(str(Path(input_path).resolve()))
            deck.SaveAs(str(Path(output_pdf).resolve()), 32)  # 32 = PDF
            deck.Close()
            powerpoint.Quit()

            return output_pdf
        except Exception:
            pass

        # 2️⃣ Try LibreOffice
        try:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", output_dir, input_path],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return output_pdf
        except Exception:
            pass

        raise ValueError("Unable to convert PPT/PPTX to PDF")

    def _convert_excel_to_pdf(self, file_path: str = None, input_bytes=None) -> str:
        """
        Convert XLS/XLSX to PDF using:
        1. Excel COM (Windows)
        2. LibreOffice
        """

        # write bytes if needed
        if input_bytes is not None:
            original_name = getattr(input_bytes, "name", "uploaded.xlsx")
            ext = os.path.splitext(original_name)[1] or ".xlsx"
            temp_input_path = tempfile.mktemp(suffix=ext)

            if hasattr(input_bytes, "read"):
                input_bytes.seek(0)
                content = input_bytes.read()
            else:
                content = input_bytes

            with open(temp_input_path, "wb") as f:
                f.write(content)

            input_path = temp_input_path

        elif file_path:
            input_path = file_path

        else:
            raise ValueError("Must supply either file_path or input_bytes")

        output_dir = tempfile.mkdtemp()
        output_pdf = os.path.join(output_dir, Path(input_path).stem + ".pdf")

        # 1️⃣ Try Excel COM (Windows)
        try:
            import win32com.client
            excel = win32com.client.Dispatch("Excel.Application")
            excel.Visible = False

            wb = excel.Workbooks.Open(str(Path(input_path).resolve()))
            wb.ExportAsFixedFormat(0, str(Path(output_pdf).resolve()))  # 0 = PDF
            wb.Close()
            excel.Quit()

            return output_pdf
        except Exception:
            pass

        # 2️⃣ Try LibreOffice
        try:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", output_dir, input_path],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return output_pdf
        except Exception:
            pass

        raise ValueError("Unable to convert XLS/XLSX to PDF")


