import json
import os
import tempfile
import base64

from langchain.chat_models import init_chat_model
from openai import OpenAI
from PIL import Image


from dotenv import load_dotenv
from openai.types import ChatModel

from .image_processor import ImageProcessor


load_dotenv(override=True)


class DocuToMarkdownExtractor:
    """Sends image pages to an LLM and extracts Markdown text + tables."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",client:ChatModel=None):
        if client is None:
            client = init_chat_model(
                model=model,
                model_provider="openai",  # you can later swap to "anthropic", "google", etc.
                api_key=api_key
            )
        self.client = client
        self.model = client.model_name
        # Initialize ImageProcessor once and pass the chat model
        self.processor = ImageProcessor(client=self.client)

    

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts PIL image to base64-encoded PNG string."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            with open(tmp.name, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    def extract_markdown_pages(self, pages, include_image: bool = True):
        """Extracts Markdown from page images and integrates embedded image data."""
        all_outputs = []
        text_content = ""

        for i, page in enumerate(pages, start=1):
            print(f"🧠 Processing page {i}/{len(pages)}...")

            # 1. Get the primary page image for OCR/Parsing
            # If 'page' is a dict, we use the 'rendered_image' key
            rendered_image = page.get("page_image")
            b64_page_image = self._image_to_base64(rendered_image)

            processor = ImageProcessor(model_name="gpt-4o-mini")

            fins = [
                {"type": "text", "text": (
                    "You are a document parser. Extract all text, images and tables "
                    "from this image and format the output in clean Markdown. "
                    "Preserve table structure, headings, and lists. If no markdown, put a space. "
                    "Describe any visual elements or images found on this page."
                    "Return only VALID JSON with keys: markdown_text, short_title, page_number, summary."
                )},

            ]

            response = processor.analyze(encoded_image=b64_page_image, finstructioncontent=fins)

            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    print('skipping quietly')
                    continue

            # 2. Integrate text content
            text_content += "\n" + response.get("markdown_text", "")

            # 3. Handle Embedded Images for this page
            # We attach the high-res embedded images found in the DOCX to the page response
            if include_image:
                response["page_image_data"] = b64_page_image
                # Map the specific embedded images extracted from the DOCX for this page

                response["embedded_images"] = self.conv_to_base64(page,response)

            response["image_index"] = i
            response["page_number"] = i

            all_outputs.append(response)

        return all_outputs, text_content

    def conv_to_base64(self, page,response):

        # Get the raw list of embedded images (which likely contains binary blobs)
        raw_embedded_list = page.get("embedded_images", [])
        encoded_images = []

        for img in raw_embedded_list:
            # Check if we have image_bytes or a blob
            # The source data from your extractor typically provides 'image_bytes' or 'blob'
            image_bytes = img.get("image_bytes") or img.get("image_data")

            if isinstance(image_bytes, bytes):
                # Encode to base64 and decode to utf-8 string for JSON compatibility
                b64_string = base64.b64encode(image_bytes).decode("utf-8")
            else:
                # If it's already a string or empty, keep as is
                b64_string = image_bytes

            encoded_images.append({
                "image_index": img.get("image_index"),
                "image_data": b64_string,  # Now a Base64 string
                "content_type": img.get("content_type", "image/png")
            })

        return json.dumps(encoded_images)
    def extract_markdown(self, images,include_image:bool=True):
        """Extracts Markdown-formatted text from each image page."""
        all_outputs = []
        text_content=""
        for i, image in enumerate(images, start=1):
            print(f"🧠 Processing page {i}/{len(images)}...")

            b64_image = self._image_to_base64(image)
            processor = ImageProcessor(model_name="gpt-4o-mini")

            fins = [{"type": "text", "text": "You are a document parser. Extract all text, images and tables "
                                             "from this image and format the output in clean Markdown. "
                                             "Preserve table structure, headings, and lists. If there is no markdown, put a space. "
                                             "Put your result in a JSON object with the following keys:"
                                             "- markdown_text: the markdown text"
                                             "- short_title: the short title of the document"
                                             "- page_number: the page number of the document (i+1)"
                                             "- summary: a summary of the document,"
                                             " - image_data: the image data in base64 format,"
                                             "Return only raw JSON, without markdown formatting or triple backticks."
                                             "- image_index: the index of the image in the document"},
                    {"type": "text", "text": "You are an image inspector. Tell us what is in the image "
                                             "or what the document is about."},
                    ]
            response=processor.analyze(encoded_image=b64_image, finstructioncontent=fins)

            if isinstance(response, str):
                try:
                    response = json.loads(response)  # Convert JSON string to dictionary
                except json.JSONDecodeError:
                    print('skipping quietly')
                    #raise ValueError("The response from 'processor.analyze' is not valid JSON.")
            text_content=text_content+"\n"+response["markdown_text"]
            if(include_image):
                response["image_data"]=b64_image
            response["image_index"]=i
            response["page_number"] = i

            all_outputs.append(response)

        json_array = json.dumps(all_outputs, indent=2)
        print(json_array)
        return all_outputs, text_content

