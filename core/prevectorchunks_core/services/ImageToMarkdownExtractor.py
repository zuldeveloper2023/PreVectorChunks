
import os
import tempfile
import base64

from openai import OpenAI
from PIL import Image

from core.prevectorchunks_core.services.DocuToImageConverter import DocuToImageConverter
from dotenv import load_dotenv
load_dotenv(override=True)

# ---------------------------------------------------------
# 🧠 Class 2: Send images to LLM and extract Markdown
# ---------------------------------------------------------
class ImageToMarkdownExtractor:
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

    def extract_markdown(self, images):
        """Extracts Markdown-formatted text from each image page."""
        all_outputs = []

        for i, image in enumerate(images, start=1):
            print(f"🧠 Processing page {i}/{len(images)}...")

            b64_image = self._image_to_base64(image)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are  a document parser. Extract all text and tables "
                                    "from this page and format the output in clean Markdown. "
                                    "Preserve table structure, headings, and lists. if there is no markdown, put a space"
                                ),
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are an image inspector - Tell us what is in the image"
                                    "or what is the document about"
                                ),
                            },
                            {
                                "type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                        ],
                    }
                ],
            )

            markdown_text = response.choices[0].message.content.strip()
            all_outputs.append(f"# Page {i}\n\n{markdown_text}\n\n")

        return "\n".join(all_outputs)

if __name__ == "__main__":
    converter = DocuToImageConverter()
    extractor = ImageToMarkdownExtractor(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Convert document → images
    images = converter.convert_to_images("11844470.jpg")

    # Step 2: Extract Markdown from images
    markdown_output = extractor.extract_markdown(images)

    # Step 3: Save the final result
    with open("output.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)

    print("✅ Markdown extraction complete! See output.md")

