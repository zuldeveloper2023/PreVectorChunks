# 📚 PreVectorChunks

> A lightweight utility for **document chunking** and **vector database upserts** — designed for developers building **RAG (Retrieval-Augmented Generation)** solutions.

---

## ✨ Who Needs This Module?
Any developer working with:
- **RAG pipelines**
- **Vector Databases** (like Pinecone, Weaviate, etc.)
- **AI applications** requiring **similar content retrieval**

---

## 🎯 What Does This Module Do?
This module helps you:
- **Chunk documents** into smaller fragments using:
  - a pretrained Reinforcement Learning based model or
  - a pretrained Reinforcement Learning based model with proposition indexing or
  - standard word chunking
  - recursive character based chunking
  - character based chunking
- **Insert (upsert) fragments** into a vector database  
- **Fetch & update** existing chunks from a vector database  

---

## 📦 Installation
```bash
pip install prevectorchunks-core
```

How to import in a file:  
```python
from PreVectorChunks.services import chunk_documents_crud_vdb
```

**Use .env for API keys:IMPORTANT: PLEASE ENSURE TO PROVIDE YOUR OPENAI_API_KEY as MINIMUM in an .env file or as required**
```
PINECONE_API_KEY=YOUR_API_KEY
OPENAI_API_KEY=YOUR_API_KEY
```

---

## 📄 Functions

### 1. `chunk_documents`
```python
chunk_documents(instructions, file_path="content_playground/content.json", splitter_config=SplitterConfig())
```
Splits the content of a document into smaller, manageable chunks. - Five types of document chunking
- Chunking using Reinforcement Learning based pretrained model +(enable/disable LLM to structure the chunked text - default is enabled)
- Chunking using Reinforcement Learning based pretrained model and proposition indexing +(enable/disable LLM to structure the chunked text - default is enabled)
- Recursive Character based chunking +(enable/disable LLM to structure the chunked text - default is enabled)
- Standard word based chunking+(enable/disable LLM to structure the chunked text - default is enabled)
- Simple character based chunking +(enable/disable LLM to structure the chunked text - default is enabled)


**Parameters**
- `instructions` (*dict or str*): Additional rules or guidance for how the document should be split.  
  - Example: `"split my content by biggest headings"`
- `file_path` (*str*): Binary file or file path to the input file containing the content or content of the file. Default: `"content_playground/content.json"`.
- `splitter_config (optional) ` (*SplitterConfig*): (if none provided standard split takes place) Object that defines chunking behavior, e.g., `chunk_size`, `chunk_overlap`, `separator`, `split_type`.
- i.e. splitter_config = SplitterConfig(chunk_size= 300, chunk_overlap= 0,separators=["\n"],split_type=SplitType.RECURSIVE.value)
- (chunk_size refers to size in characters (i.e. 100 characters) when RECURSIVE is used)
- i.e. splitter_config = SplitterConfig(chunk_size= 300, chunk_overlap= 0,separators=["\n"],split_type=SplitType.CHARACTER.value)
- - (chunk_size refers to size in characters (i.e. 100 characters) when CHARACTER is used)
- i.e. splitter_config = SplitterConfig(chunk_size= 300, chunk_overlap= 0,separators=["\n"],split_type=SplitType.STANDARD.value)
- - (chunk_size refers to size in words (i.e. 100 characters) when STANDARD is used)
- i.e. splitter_config = SplitterConfig(separators=["\n"],
                                     split_type=SplitType.R_PRETRAINED.value, min_rl_chunk_size=5,
                                     max_rl_chunk_size=50,enableLLMTouchUp=False)
- - (min_rl_chunk_size and max_rl_chunk_size refers to size in sentences (i.e. 100 sentences) when R_PRETRAINED is used)
- i.e. splitter_config = SplitterConfig(separators=["\n"],
                                     split_type=SplitType.R_PRETRAINED_PROPOSITION.value, min_rl_chunk_size=5,
                                     max_rl_chunk_size=50,enableLLMTouchUp=False)
- - (min_rl_chunk_size and max_rl_chunk_size refers to size in sentences (i.e. 100 sentences) when R_PRETRAINED_PROPOSITION is used)
**Returns**
- A list of chunked strings including a unique id, a meaningful title and chunked text

**Use Cases**
- Preparing text for LLM ingestion
- Splitting text by structure (headings, paragraphs)
- Vector database indexing

---

### 2. `chunk_and_upsert_to_vdb`
```python
chunk_and_upsert_to_vdb(index_n, instructions, file_path="content_playground/content.json", splitter_config=SplitterConfig())
```
Splits a document into chunks (via `chunk_documents`) and **inserts them into a Vector Database**.

**Parameters**
- `index_n` (*str*): The name of the VDB index where chunks should be stored.
- `instructions` (*dict or str*): Rules for splitting content (same as `chunk_documents`).
- `file_path` (*str*): Path to the document file or content of the file. Default: `"content_playground/content.json"`.
- `splitter_config` (*SplitterConfig*): Object that defines chunking behavior.

**Returns**
- Confirmation of successful insert into the VDB.

**Use Cases**
- Automated document preprocessing and storage for vector search
- Preparing embeddings for semantic search

---

### 3. `fetch_vdb_chunks_grouped_by_document_name`
```python
fetch_vdb_chunks_grouped_by_document_name(index_n)
```
Fetches existing chunks stored in the Vector Database, grouped by **document name**.

**Parameters**
- `index_n` (*str*): The name of the VDB index.

**Returns**
- A dictionary or list of chunks grouped by document name.

**Use Cases**
- Retrieving all chunks of a specific document
- Verifying what content has been ingested into the VDB

---

### 4. `update_vdb_chunks_grouped_by_document_name`
```python
update_vdb_chunks_grouped_by_document_name(index_n, dataset)
```
Updates existing chunks in the Vector Database by document name.

**Parameters**
- `index_n` (*str*): The name of the VDB index.  
- `dataset` (*dict or list*): The new data (chunks) to update existing entries.

**Returns**
- Confirmation of update status.

**Use Cases**
- Keeping VDB chunks up to date when documents change
- Re-ingesting revised or corrected content

---

## 🚀 Example Workflow
```python
from prevectorchunks_core.config import SplitterConfig

splitter_config = SplitterConfig(chunk_size=150, chunk_overlap=0, separator=["\n"], split_type=SplitType.R_PRETRAINED_PROPOSITION.value)

# Step 1: Chunk a document
chunks = chunk_documents(
    instructions="split my content by biggest headings",
    file_path="content_playground/content.json",
    splitter_config=splitter_config
)

splitter_config = SplitterConfig(chunk_size=300, chunk_overlap=0, separators=["\n"],
                                     split_type=SplitType.R_PRETRAINED_PROPOSITION.value, min_rl_chunk_size=5,
                                     max_rl_chunk_size=50,enableLLMTouchUp=False)

chunks=chunk_documents_crud_vdb.chunk_documents("extract", file_name=None, file_path="content.txt",splitter_config=splitter_config)

# Step 2: Insert chunks into VDB
chunk_and_upsert_to_vdb("my_index", instructions="split by headings", splitter_config=splitter_config)

# Step 3: Fetch stored chunks
docs = fetch_vdb_chunks_grouped_by_document_name("my_index")

# Step 4: Update chunks if needed
update_vdb_chunks_grouped_by_document_name("my_index", dataset=docs)
```

---

## 🛠 Use Cases
- Preprocessing documents for LLM ingestion  
- Semantic search and Q&A systems  
- Vector database indexing and retrieval  
- Maintaining versioned document chunks

