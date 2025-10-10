"""
propositional_indexer.py
A scalable LangChain-based propositional indexing pipeline.
Uses LangChain Hub template "wfh/proposal-indexing" with batching support.
"""
from dotenv import load_dotenv

from ..utils.extract_content import extract_content_agnostic
load_dotenv(override=True)# must come firs
# t
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel
from typing import List
import os
from langchain_openai import ChatOpenAI
print(os.getenv("OPENAI_API_KEY"))
# -----------------------------
# 1️⃣ Define data model
# -----------------------------
class Sentences(BaseModel):
    sentences: List[str]

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel
from typing import List

class Sentences(BaseModel):
    sentences: List[str]

class PropositionalIndexer:
    """
    Class-based wrapper for LangChain Hub propositional indexing.
    Handles model, key, and batching automatically.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", max_chars: int = 4000):
        # Load environment variables
        load_dotenv(override=True)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env or environment!")

        self.model_name = model_name
        self.max_chars = max_chars

        # Initialize LLM and chain
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            openai_api_key=self.api_key
        )
        self.chain = self._load_chain()

    def _load_chain(self):
        """Load LangChain Hub propositional indexing template"""
        obj = hub.pull("wfh/proposal-indexing")
        extraction_llm = self.llm.with_structured_output(Sentences)
        return obj | extraction_llm

    def _chunk_text(self, text: str):
        """Split text into batches for large documents"""
        chunks = []
        current, current_len = [], 0
        for paragraph in text.split("\n"):
            if current_len + len(paragraph) > self.max_chars:
                chunks.append("\n".join(current))
                current = [paragraph]
                current_len = len(paragraph)
            else:
                current.append(paragraph)
                current_len += len(paragraph)
        if current:
            chunks.append("\n".join(current))
        return chunks

    def index_text(self, text: str) -> List[str]:
        """Run propositional indexing on a text string"""
        chunks = self._chunk_text(text)
        all_sentences = []
        for i, chunk in enumerate(chunks, 1):
            print(f"➡️ Processing batch {i}/{len(chunks)} ({len(chunk)} chars)")
            try:
                result = self.chain.invoke(chunk)
                all_sentences.extend([s.strip() for s in result.sentences])
            except Exception as e:
                print(f"⚠️ Error on batch {i}: {e}")
        return all_sentences

    def index_file(self, input_path_or_binary, output_path: str,file_name=None) -> List[str]:
        """Run propositional indexing on a text file and save results"""
        text=extract_content_agnostic(input_path_or_binary,file_name)

        sentences = self.index_text(text)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        print(f"\n✅ Indexed {len(sentences)} sentences saved to {output_path}")
        return sentences

    def index_file_content(self, content, output_path: str) -> List[str]:
        """Run propositional indexing on a text file and save results"""

        sentences = self.index_text(content)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        print(f"\n✅ Indexed {len(sentences)} sentences saved to {output_path}")
        return sentences