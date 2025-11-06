import json
import pytest

from core.prevectorchunks_core.config.splitter_config import SplitterConfig, LLM_Structured_Output_Type
from core.prevectorchunks_core.services import chunk_documents_crud_vdb
from core.prevectorchunks_core.services.markdown_and_chunk_documents import MarkdownAndChunkDocuments
from core.prevectorchunks_core.utils.file_loader import SplitType


# Create a temporary JSON file to test with
@pytest.fixture
def temp_json_file(tmp_path):
    file_path = tmp_path / "test.json"
    content = [{"id": 1, "text": "hello world"}]
    with open(file_path, "w") as f:
        json.dump(content, f)
    return file_path


def test_load_file_and_upsert_chunks_to_vdb(temp_json_file):
    splitter_config = SplitterConfig(chunk_size=300, chunk_overlap=0, separators=["\n"],
                                     split_type=SplitType.RECURSIVE.value, min_rl_chunk_size=5,
                                     max_rl_chunk_size=50, enableLLMTouchUp=True,llm_structured_output_type=LLM_Structured_Output_Type.STANDARD)

    chunks = chunk_documents_crud_vdb.chunk_documents("extract", file_name=None, file_path="content.txt",

                                                      splitter_config=splitter_config)

    print(chunks)
    for i, c in enumerate(chunks):
        print(f"Chunk {i + 1}: {c}")
    print(chunks)

def test_markdown(temp_json_file):
    markdown_and_chunk_documents = MarkdownAndChunkDocuments()
    mapped_chunks = markdown_and_chunk_documents.markdown_and_chunk_documents(
        "content.docx",include_image=True)
    print(mapped_chunks)
    for i, c in enumerate(mapped_chunks):
        print(f"Chunk {i + 1}: {c}")

    for i, c in enumerate(mapped_chunks):
        print(f"Chunk {i + 1}: {c}")
    print(mapped_chunks)
