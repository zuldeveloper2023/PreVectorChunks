import pytest
import sys
import os

from core.prevectorchunks_core.rlchunker.inference import RLChunker
from core.prevectorchunks_core.services.propositional_index import  PropositionalIndexer

# Add parent directory (project root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
# Define file paths
input_path = "content.txt"           # your input text file
output_path = "propositional_index.txt"  # where the indexed output will be saved
def split_sentences(text):
    # simple sentence splitter based on punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # clean up any empty or very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences
@pytest.fixture
def sample_text():
    return (
      ""    )




def test_chunk_text_with_pretrained_model(sample_text):
    """Test chunking a sample text"""
    file_path = "content.txt"  # your text file path here
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open("content.txt", "rb") as f:
        binary_data = f.read()

    indexer = PropositionalIndexer(model_name="gpt-4o-mini")

    # Index directly from file
    sentences = indexer.index_file(binary_data, "propositional_index.txt",file_name="content.txt")

    # ✅ Combine all sentences into one big text
    combined_text = " ".join(sentences)

    # Optionally save to another file
    with open("combined_sentences.txt", "w", encoding="utf-8") as f:
        f.write(combined_text)



    # Initialize chunker once
    chunker = RLChunker(device="cpu", embedding_dim=384)

    # Chunk a single text

    chunks = chunker.chunk_text(combined_text)

    print("Chunks:", chunks)





    for i, c in enumerate(chunks):
        print(f"Chunk {i + 1}: {c}")

   # assert isinstance(chunks, list), "Chunks should be a list"
    #assert len(chunks) > 0, "No chunks were produced"
    #assert isinstance(chunks[0], list), "Each chunk should"
