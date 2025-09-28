Who will need this module?
Any developer who uses RAG and Vector Database to find similar content as part of their AI solutions

What is this module for?
This module helps developers who wants to chunk document for the purposes of inserting fragments of texts into a vector database

how to import in a python module
pip install prevectorchunks

how to import in a file
from PreVectorChunks.services import chunk_documents_crud_vdb

how to use PineCone and OpenAI
use a .env file to be able to start chunking and storing into pinecone
PINECONE_API_KEY=YOUR API KEY
OPENAI_API_KEY=YOUR API KEY

how to call relevant functions

Four key functions that you can call are below:


#function that chunks any document
chunk_documents(instructions,file_path="content_playground/content.json"):
    

#function that chunks any document as well as inserts into vdb - you need an index name inside index_n
chunk_and_upsert_to_vdb(index_n,instructions,file_path="content_playground/content.json"):
  

#function that loads existing chunks from vdb by document name - you need an index name inside index_n
fetch_vdb_chunks_grouped_by_document_name(index_n):
    

#function that updates existing chunks - you need an index name inside index_n
update_vdb_chunks_grouped_by_document_name(index_n,dataset):


    
