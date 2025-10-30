"""
image_processor.py
A scalable LangChain-based image reasoning pipeline.
Uses OpenAI GPT-4o (multimodal) via LangChain.
"""

import base64
import os
import requests
from dotenv import load_dotenv
from typing import Optional

from openai import OpenAI
from langchain_core.pydantic_v1 import BaseModel


# -----------------------------------------------------
# 1️⃣ Define structured model for output
# -----------------------------------------------------
class ImageAnalysis(BaseModel):
    description: str
    objects_detected: Optional[list[str]] = None
    reasoning: Optional[str] = None


# -----------------------------------------------------
# 2️⃣ Image Processor class
# -----------------------------------------------------
class ImageProcessor:
    """
    Wrapper for a GPT-4o multimodal image reasoning pipeline.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        load_dotenv(override=True)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env or environment!")

        # Initialize multimodal client
        self.llm = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    # -------------------------------------------------
    # 3️⃣ Image encoding helper
    # -------------------------------------------------
    def _encode_image(self, file_path_or_url: str):
        """Return base64-encoded image string."""
        if file_path_or_url.startswith("http"):
            response = requests.get(file_path_or_url)
            response.raise_for_status()
            data = response.content
        else:
            with open(file_path_or_url, "rb") as f:
                data = f.read()

        return base64.b64encode(data).decode("utf-8")

    # -------------------------------------------------
    # 4️⃣ Analyze image
    # -------------------------------------------------
    def analyze(self, encoded_image, finstructioncontent: list[str] = "Describe the image in detail."):
        """Send an image to GPT-4o for reasoning and structured output."""
        print("🖼️ Sending image for analysis...")
        content1 = [

                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ]
        content1.extend(finstructioncontent)
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content1
                }
            ],
        )

        result_text = response.choices[0].message.content
        print("✅ Analysis complete.")
        print(result_text)
        return result_text


# -----------------------------------------------------
# 5️⃣ Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    processor = ImageProcessor()

    # Example: Local or remote image file
    image_file = "example.jpg"  # or URL: "https://example.com/pic.jpg"

    analysis = processor.analyze(
        image_file,
        instruction="List all visible objects, estimate their relationships, and summarize the scene."
    )

    print("\n--- IMAGE ANALYSIS RESULT ---")
    print(analysis)
