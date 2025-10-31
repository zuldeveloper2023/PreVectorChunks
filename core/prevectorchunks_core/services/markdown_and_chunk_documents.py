import json
import os
import tempfile
import base64

from openai import OpenAI
from PIL import Image
from .DocuToImageConverter import DocuToImageConverter

from .DocuToMarkdownExtractor import DocuToMarkdownExtractor
from ..config.splitter_config import SplitterConfig

from dotenv import load_dotenv

from .chunk_documents_crud_vdb import chunk_documents
from .chunk_to_all_content_mapper import ChunkMapper
from ..utils.file_loader import SplitType

load_dotenv(override=True)


class MarkdownAndChunkDocuments:


    def markdown_and_chunk_documents(self,file_path:str):
        # Create instances of the converter and extractor
        converter = DocuToImageConverter()
        extractor = DocuToMarkdownExtractor(api_key=os.getenv("OPENAI_API_KEY"))


        images = converter.convert_to_images(file_path)

        # convert
        # Step 2: Extract Markdown from images
        markdown_output, text_content = extractor.extract_markdown(images, include_image=False)
        # convert text content to binary
        binary_text_content = text_content.encode('utf-8')  # bytes representation

        chunk_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        cm = ChunkMapper(chunk_client, markdown_output, embedding_model="text-embedding-3-small")
        splitter_config = SplitterConfig(chunk_size=300, chunk_overlap=0, separators=["\n"],
                                         split_type=SplitType.R_PRETRAINED_PROPOSITION.value, min_rl_chunk_size=5,
                                         max_rl_chunk_size=50, enableLLMTouchUp=False)

        chunked_text = chunk_documents("", file_name="install_ins.txt", file_path=binary_text_content,
                                       splitter_config=splitter_config)

        flat_chunks = result = [''.join(inner) for inner in chunked_text]
        mapped_chunks = cm.map_chunks(flat_chunks)
        for md_item in markdown_output:
            # Check if this markdown_output item is already present in mapped_chunks
            match_found = False
            for mapped in mapped_chunks:
                if mapped.get("markdown_text") == md_item.get("markdown_text"):
                    match_found = True
                    break

            # If not found, append the missing markdown_output item
            if not match_found:
                md_item["chunked_text"] = md_item["markdown_text"]
                mapped_chunks.append(md_item)
        #print(mapped_chunks)

        #print("✅ Markdown extraction complete! See output.md")
        return mapped_chunks


if __name__ == "__main__":
    markdown_and_chunk_documents = MarkdownAndChunkDocuments()
    mapped_chunks=markdown_and_chunk_documents.markdown_and_chunk_documents("421307-nz-au-top-loading-washer-guide-shorter.pdf")
    print(mapped_chunks)