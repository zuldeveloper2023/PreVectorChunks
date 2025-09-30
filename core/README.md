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
- **Chunk documents** into smaller fragments  
- **Insert (upsert) fragments** into a vector database  
- **Fetch & update** existing chunks from a vector database  

---

## 📦 Installation
```bash
pip install prevectorchunks-core
````
How to import in a file:  
```python
from PreVectorChunks.services import chunk_documents_crud_vdb

#How to use Pinecone and OpenAI:
#Use a .env file in your project root to configure API keys:

PINECONE_API_KEY=YOUR_API_KEY
OPENAI_API_KEY=YOUR_API_KEY

#how to call relevant functions:
#Four key functions that you can call are below: 
#function that chunks any document 
```

---

## 📄 Functions

### 1. `chunk_documents`
```python
chunk_documents(instructions, file_path="content_playground/content.json", chunk_size=200)
```
Splits the content of a document into smaller, manageable chunks.

**Parameters**
- `instructions` (*dict or str*): Additional rules or guidance for how the document should be split.  
  - Example: `"split my content by biggest headings"`
- `file_path` (*str*): Path to the input JSON/text file containing the content or content of the file. Default: `"content_playground/content.json"`.
- `chunk_size` (*int*): Number of words per chunk (default = 200).

**Returns**
- A list of chunked strings including a unique id, a meaningful title and chunked text

**Use Cases**
- Preparing text for LLM ingestion
- Splitting text by structure (headings, paragraphs)
- Vector database indexing

---

### 2. `chunk_and_upsert_to_vdb`
```python
chunk_and_upsert_to_vdb(index_n, instructions, file_path="content_playground/content.json", chunk_size=200)
```
Splits a document into chunks (via `chunk_documents`) and **inserts them into a Vector Database**.

**Parameters**
- `index_n` (*str*): The name of the VDB index where chunks should be stored. for example, in pinecone, we can have an index 'dl-doco'
- `instructions` (*dict or str*): Rules for splitting content (same as `chunk_documents`).
- `file_path` (*str*): Path to the document file or content of the file. Default: `"content_playground/content.json"`.
- `chunk_size` (*int*): Max words per chunk. Default: `200`.

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
# Step 1: Chunk a document
chunks = chunk_documents(
    instructions="split my content by biggest headings",
    file_path="content_playground/content.json",
    chunk_size=150
)

# Step 2: Insert chunks into VDB
chunk_and_upsert_to_vdb("my_index", instructions="split by headings")

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

