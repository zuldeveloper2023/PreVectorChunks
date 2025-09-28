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
pip install prevectorchunks

How to import in a file:  
```python
from PreVectorChunks.services import chunk_documents_crud_vdb

How to use Pinecone and OpenAI:
Use a .env file in your project root to configure API keys:

PINECONE_API_KEY=YOUR_API_KEY
OPENAI_API_KEY=YOUR_API_KEY

how to call relevant functions Four key functions that you can call are below: 
#function that chunks any document 
chunk_documents(instructions,file_path="content_playground/content.json"): 
#function that chunks any document as well as inserts into vdb - you need an index name inside index_n
chunk_and_upsert_to_vdb(index_n,instructions,file_path="content_playground/content.json"): 
#function that loads existing chunks from vdb by document name - you need an index name inside index_n 
fetch_vdb_chunks_grouped_by_document_name(index_n): 
#function that updates existing chunks - you need an index name inside index_n 
update_vdb_chunks_grouped_by_document_name(index_n,dataset):