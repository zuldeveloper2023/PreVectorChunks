import json
import os
import tempfile
import base64

from openai import OpenAI
from PIL import Image

from core.prevectorchunks_core.config.splitter_config import SplitterConfig
from core.prevectorchunks_core.services.DocuToImageConverter import DocuToImageConverter
from dotenv import load_dotenv

from core.prevectorchunks_core.services.chunk_documents_crud_vdb import chunk_documents
from core.prevectorchunks_core.services.chunk_to_all_content_mapper import ChunkMapper
from core.prevectorchunks_core.services.image_processor import ImageProcessor
from core.prevectorchunks_core.utils.file_loader import SplitType

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
                                             "Put your result in a JSON object with the following keys:\n"
                                             "- markdown_text: the markdown text\n"
                                             "- short_title: the short title of the document\n"
                                             "- page_number: the page number of the document (image index + 1)\n"
                                             "- summary: a summary of the document\n,"
                                             " - image_data: the image data in base64 format\n,"
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
            all_outputs.append(response)

        json_array = json.dumps(all_outputs, indent=2)
        print(json_array)
        return all_outputs, text_content


if __name__ == "__main__":
    # Create instances of the converter and extractor
    converter = DocuToImageConverter()
    extractor = DocuToMarkdownExtractor(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Convert document → images
    file_path = "421307-nz-au-top-loading-washer-guide-shorter.pdf"  # Replace with your file path
    images = converter.convert_to_images(file_path)

    #convert
    # Step 2: Extract Markdown from images
    markdown_output,text_content = extractor.extract_markdown(images,include_image=False)
    # convert text content to binary
    binary_text_content= text_content.encode('utf-8')  # bytes representation


    chunk_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cm= ChunkMapper(chunk_client,markdown_output,embedding_model="text-embedding-3-small")
    splitter_config = SplitterConfig(chunk_size=300, chunk_overlap=0, separators=["\n"],
                                     split_type=SplitType.R_PRETRAINED_PROPOSITION.value, min_rl_chunk_size=5,
                                     max_rl_chunk_size=50, enableLLMTouchUp=True)



    chunked_text=chunk_documents("",file_name="install_ins.txt",file_path=binary_text_content,splitter_config=splitter_config)
    flat_chunks = [item for sublist in chunked_text for item in sublist]
    mapped_chunks=cm.map_chunks(flat_chunks)
    print(mapped_chunks)

    print("✅ Markdown extraction complete! See output.md")
