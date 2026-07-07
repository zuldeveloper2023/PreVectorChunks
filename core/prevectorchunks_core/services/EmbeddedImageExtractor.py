import os
import tempfile
import shutil
import subprocess
from base64 import b64encode
from pathlib import Path
from PIL import Image
import io
import fitz
from docx2pdf import convert as docx2pdf_convert
from docx import Document


class EmbeddedImageExtractor:
    """Converts a document (PDF, DOCX, DOC, image bytes) into a list of PIL images."""

    def __init__(self):
        pass

    # ----------------------------
    # DOCX helper
    # ----------------------------
    def extract_all_images_from_docx(self, file_path, page_number):
        doc = Document(file_path)
        images = []

        # Method A: Standard Relationship check
        for rId, rel in doc.part.related_parts.items():
            if "image" in rel.content_type:
                image_bytes = rel.blob
                images.append(self._format_output(image_bytes, rId))

        # Method B: Package Part check (The "Deep Dive")
        # If Method A found nothing, we look at every part in the zip package
        if not images:
            for part in doc.part.package.parts:
                if "image" in part.content_type:
                    images.append(self._format_output(part.blob, "unknown_rid"))

        return images



    def extract_embedded_images_from_pdf_page(self, file_path, page_number):
        """
        Extract all embedded images from a PDF.

        Returns:
        [
            {
                "page": 1,
                "image_index": 1,
                "image_data": "...base64...",
                "width": 640,
                "height": 480,
                "ext": "png"
            },
            ...
        ]
        """
        images = []

        pdf = fitz.open(file_path)

        try:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)

                # List all images on this page
                image_list = page.get_images(full=True)

                for image_index, img in enumerate(image_list, start=1):
                    xref = img[0]

                    image_info = pdf.extract_image(xref)

                    image_bytes = image_info["image"]

                    images.append({
                        "page": page_num + 1,
                        "image_index": image_index,
                        "image_data": b64encode(image_bytes).decode("utf-8"),
                        "width": image_info.get("width"),
                        "height": image_info.get("height"),
                        "ext": image_info.get("ext")
                    })

        finally:
            pdf.close()

            return images
    def _format_output(self, blob, rId):
        return {
            "image_data": b64encode(blob).decode("utf-8"),
            "rel_id": rId
        }

