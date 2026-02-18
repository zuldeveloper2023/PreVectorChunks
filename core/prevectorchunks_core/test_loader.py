import json
import pytest
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from core.prevectorchunks_core.config.splitter_config import SplitterConfig, LLM_Structured_Output_Type
from core.prevectorchunks_core.services import chunk_documents_crud_vdb
from core.prevectorchunks_core.services.markdown_and_chunk_documents import MarkdownAndChunkDocuments
from core.prevectorchunks_core.utils.file_loader import SplitType
import os
load_dotenv(override=True)
# Create a temporary JSON file to test with
@pytest.fixture
def temp_json_file(tmp_path):
    file_path = tmp_path / "test.json"
    content = [{"id": 1, "text": "hello world"}]
    with open(file_path, "w") as f:
        json.dump(content, f)
    return file_path


def test_load_file_and_upsert_chunks_to_vdb():
    splitter_config = SplitterConfig(chunk_size=300, chunk_overlap=0, separators=["\n"],
                                     split_type=SplitType.R_PRETRAINED_PROPOSITION.value, min_rl_chunk_size=5,
                                     max_rl_chunk_size=50, enableLLMTouchUp=True,llm_structured_output_type=LLM_Structured_Output_Type.STRUCTURED_WITH_VECTOR_DB_ID_GENERATED)
    client = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",  # you can later swap to "anthropic", "google", etc.
        api_key=os.getenv("OPENAI_API_KEY")
    )
    chunks = chunk_documents_crud_vdb.chunk_documents("extract", file_name=None, file_path="C:\\test-sandbox\\be\\PreVectorDeps\\PreVectorChunks\\core\\prevectorchunks_core\\services\\content.pptx",

                                                      splitter_config=splitter_config,client=client)

    print(chunks)
    for i, c in enumerate(chunks):
        print(f"Chunk {i + 1}: {c}")
    print(chunks)

def test_markdown():

    client = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",  # you can later swap to "anthropic", "google", etc.
        api_key=os.getenv("OPENAI_API_KEY")
    )
    markdown_and_chunk_documents = MarkdownAndChunkDocuments(client)
    embedding_client = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    mapped_chunks = markdown_and_chunk_documents.markdown_and_chunk_documents(
        "C:\\test-sandbox\\be\\PreVectorDeps\\PreVectorChunks\\core\\prevectorchunks_core\\services\\content.pptx",include_image=True,embedding_client=embedding_client)
    print(mapped_chunks)
    for i, c in enumerate(mapped_chunks):
        print(f"Chunk {i + 1}: {c}")

    for i, c in enumerate(mapped_chunks):
        print(f"Chunk {i + 1}: {c}")
    print(mapped_chunks)
