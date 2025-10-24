"""
audio_processor.py
A scalable LangChain-based audio transcription pipeline.
Uses LangChain Hub template "wfh/audio-transcription" (example).
"""

import base64
import os
import requests
from dotenv import load_dotenv
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from typing import Optional

from openai import OpenAI


# -----------------------------------------------------
# 1️⃣ Define structured model for output
# -----------------------------------------------------
class Transcription(BaseModel):
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


# -----------------------------------------------------
# 2️⃣ Audio Processor class
# -----------------------------------------------------
class AudioProcessor:
    """
    Class-based wrapper for LangChain Hub audio transcription template.
    Fetches an audio-transcription chain from the Hub and pipes it through the LLM.
    """

    def __init__(self, model_name: str = "gpt-4o-audio-preview"):
        load_dotenv(override=True)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env or environment!")

        # Initialize LLM that supports audio modality
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        #     ChatOpenAI(
        #     model=model_name,
        #     modalities=["text", "audio"],
        #     audio={"voice": "alloy", "format": "wav"},
        #     temperature=0,
        #     openai_api_key=self.api_key,
        # ))

        # Load reusable Hub template and pipe to structured output model
        self.chain = self._load_chain()

    # -------------------------------------------------
    # 3️⃣ Internal chain loader
    # -------------------------------------------------
    def _load_chain(self):
        """
        Load LangChain Hub audio transcription template.
        (Assumes you have a template "wfh/audio-transcription" registered in Hub.)
        """
        # obj = hub.pull("wfh/audio-transcription")
        transcription_llm = self.llm
        # return obj | transcription_llm

        return transcription_llm

    # -------------------------------------------------
    # 4️⃣ Audio encoding helper
    # -------------------------------------------------
    def _encode_audio(self, file_path_or_url: str, max_size_kb: int = 100):
        """Return base64-encoded audio data. Trim if file exceeds max_size_kb."""
        if file_path_or_url.startswith("http"):
            response = requests.get(file_path_or_url)
            response.raise_for_status()
            data = response.content
        else:
            with open(file_path_or_url, "rb") as f:
                data = f.read()

        # Trim file if it exceeds the size limit


        return base64.b64encode(data).decode("utf-8")

    def _encode_audio_plain(self, file_path_or_url: str, max_size_kb: int = 100):
        """Return base64-encoded audio data. Trim if file exceeds max_size_kb."""
        if file_path_or_url.startswith("http"):
            response = requests.get(file_path_or_url)
            response.raise_for_status()
            data = response.content
        else:
            return open(file_path_or_url, "rb")

        # Trim file if it exceeds the size limit


        return data
    # -------------------------------------------------
    # 5️⃣ Transcribe method
    # -------------------------------------------------
    def transcribe(self, file_path_or_url: str, instruction: str = "Transcribe the audio recording accurately."):
        """Transcribe an audio file or URL into structured text."""
       # Prepare structured message payload (similar to OpenAI chat multimodal API)
        # Create a list of BaseMessages for compatibility
        # Check token size of input before invoking the chain
        audio_file=  open(file_path_or_url, "rb")
        transcription = self.llm.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="text",
            prompt="The following conversation is a lecture about the recent developments around OpenAI, GPT-4.5 and the future of AI."
        )

        print("🎧 Sending audio for transcription...")
        print(transcription)
        print("✅ Transcription complete.")
        return ""


# -----------------------------------------------------
# 6️⃣ Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    processor = AudioProcessor()

    # Example URL or local file
    audio_url = "13167162.mp3"

    transcription = processor.transcribe(audio_url)
    print("\n--- TRANSCRIPTION RESULT ---")

