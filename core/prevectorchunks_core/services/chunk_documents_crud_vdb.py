import json
import uuid

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
import os
from django.http import JsonResponse, HttpResponse


from ..utils.file_loader import prepare_chunked_text, extract_file_details
from ..utils.llm_wrapper import LLMClientWrapper

from pinecone import Pinecone, ServerlessSpec
from collections import defaultdict
from itertools import chain
from dotenv import load_dotenv
# create an index if not already existing
load_dotenv(override=True)
index_name = "dl-doc-search"
EMBED_DIM = 1536
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class openAIWrapperLMSContent:
    def __init__(self, openai_client):
        self.openai_client = openai_client

    def chatWithOpenAI(self, ragcontent, actualQuery):
        query = f"""
        {ragcontent}
        Question:{actualQuery}
        """

        response = self.openai_client.chat.completions.create(
            messages=[
                {'role': 'system',
                 'content': (
                     "You are an experienced and patient TAFE trainer for a Certificate III in Electrotechnology. "
                     "Your primary goal is to guide apprentices to understand their learning materials. "
                     "You must **NEVER** give direct answers to questions from assessments (UKT or UST). "
                     "Instead, use Socratic questioning to prompt critical thinking and encourage students to find the answers in their provided learning content. "
                     "When appropriate, offer hints by referencing specific sections or modules (e.g., 'Review the section on Ohm's Law in Module 2'). "
                     "Maintain a helpful, encouraging, and respectful tone. Do not solve problems for the student."
                     "{}"
                 ).format(query)}
            ],
            model="gpt-4o-mini",
            temperature=0,
        )

        return response.choices[0].message.content


# load the dataset
def loadDataset():
    dataset = load_dataset("pinecone/dl-doc-search", split="train")
    return dataset


def loadDatasetFromJsonFile(file_path="PreVectorChunks/content_playground/content.json"):
    print(os.getcwd())
    # File path to your content.json


    # Load the contents of the JSON file into a Python variable
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None


# Function to create embeddings with OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def createIndexForPineCone():
    # connect to Pinecone (make sure you set your API key in env: PINECONE_API_KEY)

    # Use the correct spec parameter
    spec = ServerlessSpec(
        cloud="aws",  # Replace with your cloud provider ('aws', 'azure', etc.)
        region="us-east-1"  # Replace with your region ('us-west-2', etc.)
    )

    # Check if the index exists. If not, create it.
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,  # Replace with the dimensionality of your vectors

            spec=spec  # Pass the spec object here
        )

    index = pc.Index(index_name)
    return index


def upsertRecord(index_n,dataset,document_name=None):
    # To get the unique host for an index,
    # see https://docs.pinecone.io/guides/manage-data/target-an-index
    index = pc.Index(index_n)

    # Upsert records into a namespace
    # `chunk_text` fields are converted to dense vectors
    # `category` fields are stored as metadata
    if dataset==None:
        dataset = loadDatasetFromJsonFile()

    try:
        rows = dataset.get("rows", [])  # Try to get "rows" from the dataset
        if rows is None:  # Check if the value is explicitly None
            rows = dataset  # Fallback to dataset
    except (AttributeError, TypeError):
        # An error occurred (e.g., `dataset` is not a dictionary or `.get()` is unavailable)
        rows = dataset  # Fallback to dataset

    # Prepare batch for upsert (do small chunks to avoid rate limits)
    vectors = []

    for i, record in enumerate(rows[:100]):  # limit to 100 for demo

        print(record)
        # Attempt to access `record["row"]["text"]` if "row" exists and is a dictionary
        text = record["row"].get("text") if isinstance(record.get("row"), dict) else record.get("text")

        title = record["row"].get("title") if isinstance(record.get("row"), dict) else record.get("title")
        id= record.get("id")
        embedding = get_embedding(text)
        value = id if id else i
        document_name_retrieved=document_name if document_name else record.get("document_name")
        vectors.append((
            str(value),  # unique ID
            embedding,
            {"title": title, "text": text,"id":id,"document_name":document_name_retrieved}  # metadata
        ))

        # Upsert in batches of 20
        if (i + 1) % 20 == 0:
            index.upsert(vectors=vectors)
            print(f"Upserted {i + 1} vectors")
            vectors = []

    # Flush remaining
    if vectors:
        index.upsert(vectors=vectors)
        print(f"Upserted final {len(vectors)} vectors")


def searchOrQueryOnVDB(queryEmbedding):
    # To get the unique host for an index,
    # see https://docs.pinecone.io/guides/manage-data/target-an-index
    index = pc.Index(index_name)

    # Search the dense index
    results = index.query(
        vector=queryEmbedding,

        top_k=10,  # Number of top results to retrieve
        include_metadata=True
    )

    # Collect top 5 results
    top_5_results = sorted(results.matches, key=lambda result: result.score, reverse=True)[:5]

    # Initialize an empty list to store the top results
    top_results_list = []

    # Append the metadata 'text' of each top result to the list
    for result in top_5_results:
        top_results_list.append(result.metadata['text'])

    # Check if there are any results
    if not top_results_list:
        print("No results found!")
    else:
        # Print out the top 5 results (if any exist)
        print("Top 5 Results:")
        for text in top_results_list:
            print(text)

    # Optionally, return the top results list
    return top_results_list


def queryToLLM(query):
    # Define the query
    # query = "the underlying principles of workplace health and safety is to:"
    query_emb = get_embedding(query)
    ragContent = searchOrQueryOnVDB(query_emb)
    system_prompt=(
                     "You are an experienced and patient TAFE trainer for a Certificate III in Electrotechnology. "
                     "Your primary goal is to guide apprentices to understand their learning materials. "
                     "You must **NEVER** give direct answers to questions from assessments (UKT or UST). "
                     "Instead, use Socratic questioning to prompt critical thinking and encourage students to find the answers in their provided learning content. "
                     "When appropriate, offer hints by referencing specific sections or copy of the section or modules (e.g., 'Review the section on Ohm's Law in Module 2'). "
                     "Maintain a helpful, encouraging, and respectful tone. Do not solve problems for the student."

                 )
    openaiwrapper = LLMClientWrapper(client=client, system_prompt=system_prompt)
    openAiResponse = openaiwrapper.chat(ragContent, query)
    return openAiResponse



# function to
# upload a particular document
# takes LLM instruction about how to process/chunk the document
# prepares chunked json objects
def upload_and_prepare_file_content_in_chunks(request,instructions,splitter_config):
    try:

        uploaded_file = uploaded_file_ref(request)
        file_name, file_bytes = extract_file_details(
            uploaded_file)
        chunked_text = chunk_documents(instructions,file_name,file_bytes,splitter_config=splitter_config)
        return chunked_text,file_name
    except Exception as e:
        return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)

# function to
# upload a particular document
def uploaded_file_ref(request):
    # Check if the request contains a file
    if 'file' not in request.FILES:
        return JsonResponse({"error": "No file part in the request"}, status=400)

    # Retrieve the file from the POST request
    uploaded_file = request.FILES['file']
    return uploaded_file

def upload_file(request):
    """
       Endpoint to handle POST requests containing a JSON file.
       The JSON file is used to load and parse data.
       """
    try:
        uploaded_file = uploaded_file_ref(request)

        if uploaded_file.name == '':
            return JsonResponse({"error": "No file selected for uploading"}, status=400)

        # Read and parse file content as JSON
        data = None
        try:
            file_data = uploaded_file.read().decode('utf-8')  # Decode file content
            data = json.loads(file_data)
            return data # Parse content as JSON
        except json.JSONDecodeError as e:
            return JsonResponse({"error": f"Failed to parse JSON: {e}"}, status=400)


    except Exception as e:
        return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def fetch_records_grouped_by_document_name(request):
    index = pc.Index(index_name)
    records_by_doc = qfetch_records_grouped_by_document_name(index)
    print(records_by_doc.keys())
    return JsonResponse(records_by_doc, safe=False)



@csrf_exempt
@require_http_methods(["POST"])
def update_records_grouped_by_document_name_in_vdb(request):
    dataset=upload_file(request)
    index_n = request.POST.get("index_name")
    transformed_records = transform_groupd_by_doc_name_to_vdb_metadata_structure(dataset)
    upsertRecord(index_n,transformed_records)
    return JsonResponse(transformed_records, safe=False)

##queries and update records within vdb

def transform_groupd_by_doc_name_to_vdb_metadata_structure(input_json):
    # Initialize an empty list to store the transformed data
    output_list = []

    # Iterate through the dictionary
    for document_name, entries in input_json.items():
        for entry in entries:
            # Add the document name to each item and append to the new list
            transformed_entry = {
                "text": entry["text"],
                "title": entry["title"],
                "id": entry["id"],
                "document_name": document_name
            }
            output_list.append(transformed_entry)

    return output_list




def qfetch_records_grouped_by_document_name(index, batch_size=100,limit=100):
    """
    Fetch all records from Pinecone grouped by document_name in metadata,
    processing 100 unique document_name values at a time.

    Args:
        index: Pinecone Index object
        batch_size: Number of unique document_name to process per batch
    Returns:
        dict: {document_name: [records]}
    """
    if limit <= 0 or limit > 100:
        raise ValueError("The `limit` parameter must be between 1 and 100.")
    # Step 1: Get all IDs with metadata
    all_ids = []
    next_token = None
    consecutive_none_count=0
    while True:
        res = list(index.list(limit=100, next_token=next_token) if next_token else index.list(limit=limit))
        if not res:
            break  # Exit loop if there are no more results

        all_ids.extend(chain.from_iterable(res))

                # Update `next_token` for pagination
        next_token = res[-1].get('next_token') if isinstance(res[-1], dict) and 'next_token' in res[-1] else None

        # Handle cases when `next_token` is None
        if not next_token:
            consecutive_none_count += 1
        else:
            consecutive_none_count = 0  # Reset counter if next_token is valid again

        # Break loop if `next_token` is None more than once consecutively
        if consecutive_none_count > 1:  # Adjust count based on logic
            break
    # Step 2: Fetch metadata in batches and collect document_names
    id_to_docname = {}
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        res = index.fetch(ids=batch_ids)
        vectors = res.vectors
        for vector in vectors.items():
            metadata = vector[1].metadata if vector[1] and vector[1].metadata else None
            doc_name = metadata['document_name'] if metadata and 'document_name' in metadata else None
            id = metadata['id'] if metadata and 'id' in metadata else None
            if doc_name and id:
                id_to_docname[id] = doc_name

    # Step 3: Group records by document_name in batches of 100 unique names
    grouped_records = defaultdict(list)
    unique_docnames = list(set(id_to_docname.values()))

    for i in range(0, len(unique_docnames), 100):
        batch_docnames = unique_docnames[i:i + 100]
        # Get all IDs for these document_names
        batch_ids = [rid for rid, dname in id_to_docname.items() if dname in batch_docnames]
        # Fetch full records
        res = index.fetch(ids=batch_ids)
        vectors = res.vectors
        for vector in vectors.items():
            metadata = vector[1].metadata if vector[1] and vector[1].metadata else None
            id = metadata['id'] if metadata and 'id' in metadata else None
            doc_name = id_to_docname[id] if id in id_to_docname else None
            text = metadata['text'] if metadata and 'text' in metadata else None
            title = metadata['title'] if metadata and 'title' in metadata else None
            id = metadata['id'] if metadata and 'id' in metadata else None

            # Create a dictionary
            json_object = {
                "text": text,
                "title": title,
                "id": id
            }
            grouped_records[doc_name].append(json_object)


    return dict(grouped_records)



#function that chunks any document
def chunk_documents(instructions,file_name,file_path="content_playground/content.json",splitter_config=None):
    return prepare_chunked_text(file_path, file_name,instructions,splitter_config=splitter_config)

#function that chunks any document as well as inserts into vdb
def chunk_and_upsert_to_vdb(index_n,instructions,file_name,file_path="content_playground/content.json",splitter_config=None):
    chunked_dataset = prepare_chunked_text(file_path, file_name, instructions,splitter_config)
    document_name = file_name if file_name else os.path.basename(file_path)   + uuid.uuid4().hex
    
    upsertRecord(index_n,chunked_dataset,document_name)
    return chunked_dataset, document_name

#function that loads existing chunks from vdb by document name
def fetch_vdb_chunks_grouped_by_document_name(index_n):
    index = pc.Index(index_n)
    records_by_doc = qfetch_records_grouped_by_document_name(index)
    return records_by_doc

#function that updates existing chunks
def update_vdb_chunks_grouped_by_document_name(index_n,dataset):
    index = pc.Index(index_n)
    transformed_records = transform_groupd_by_doc_name_to_vdb_metadata_structure(dataset)
    upsertRecord(index_n,transformed_records)
    return transformed_records
