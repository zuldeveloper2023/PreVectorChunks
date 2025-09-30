import json
import pytest

from core.prevectorchunks_core.services import chunk_documents_crud_vdb


# Create a temporary JSON file to test with
@pytest.fixture
def temp_json_file(tmp_path):
    file_path = tmp_path / "test.json"
    content = [{"id": 1, "text": "hello world"}]
    with open(file_path, "w") as f:
        json.dump(content, f)
    return file_path


def test_load_file_and_upsert_chunks_to_vdb(temp_json_file):
    #dataset = chunk_and_upsert_to_vdb("dl-doc-search","instructions", file_path="content_playground/content.json")
    #dataset=chunk_documents_crud_vdb.fetch_vdb_chunks_grouped_by_document_name("dl-doc-search")
    dataset=chunk_documents_crud_vdb.chunk_documents("extract", file_name=None, file_path=temp_json_file)
    #dataset=chunk_documents_crud_vdb.chunk_documents("Extract doco",file_path="content_playground/content.json",file_name=None)
    # Assertions

