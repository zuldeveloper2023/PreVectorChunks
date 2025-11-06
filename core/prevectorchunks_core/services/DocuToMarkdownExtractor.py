import json
import os
import tempfile
import base64

from openai import OpenAI
from PIL import Image


from dotenv import load_dotenv

from .image_processor import ImageProcessor


load_dotenv(override=True)


class DocuToMarkdownExtractor:
    """Sends image pages to an LLM and extracts Markdown text + tables."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts PIL image to base64-encoded PNG string."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            with open(tmp.name, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    def extract_markdown(self, images,include_image:True):
        """Extracts Markdown-formatted text from each image page."""
        all_outputs = []
        text_content=""
        for i, image in enumerate(images, start=1):
            print(f"🧠 Processing page {i}/{len(images)}...")

            b64_image = self._image_to_base64(image)
            processor = ImageProcessor(model_name="gpt-4o-mini")

            fins = [{"type": "text", "text": "You are a document parser. Extract all text and tables "
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
                    raise ValueError("The response from 'processor.analyze' is not valid JSON.")
            text_content=text_content+"\n"+response["markdown_text"]
            if(include_image):
                response["image_data"]=b64_image
            response["image_index"]=i
            response["page_number"] = i

            all_outputs.append(response)

        json_array = json.dumps(all_outputs, indent=2)
        print(json_array)
        return all_outputs, text_content

