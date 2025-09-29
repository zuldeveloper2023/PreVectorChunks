import json
import uuid

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openai import OpenAI
import os
from django.http import JsonResponse, HttpResponse
from PreVectorChunks.services import chunk_documents_crud_vdb

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from PreVectorChunks.utils.file_loader import extract_file_details, extract_content_agnostic

# create an index if not already existing
load_dotenv(override=True)
index_name = "dl-doc-search"
EMBED_DIM = 1536
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@csrf_exempt
@require_http_methods(["POST"])
def queryUnitContent(request):
    response = request.body
    data = json.loads(response)
    query = data["query"]
    # loaded_dataset = loadDataset()
   # index = createIndexForPineCone()
    # uprec=upsertRecord()
    response = chunk_documents_crud_vdb.queryToLLM(query)
    return HttpResponse(response)


@csrf_exempt
@require_http_methods(["POST"])
def chunk_documents_endpoint(request):
    uploaded_file=chunk_documents_crud_vdb.uploaded_file_ref(request)
    filename,file_bytes=extract_file_details(
        uploaded_file)

    return chunk_documents_crud_vdb.chunk_documents("",filename,file_bytes)


@csrf_exempt
@require_http_methods(["POST"])
def upsertContentToVDB(request):
    dataset=chunk_documents_crud_vdb.upload_file(request)
    if dataset is type(JsonResponse):
        return dataset
    else:
        chunk_documents_crud_vdb.upsertRecord(dataset)
        return JsonResponse({"message": "Data uploaded successfully"
                             ,"data":dataset}, status=200)

#
# #this service is called to:
# provide a dcoument, LLM instruction
# upload a particular document
# takes LLM instruction about how to process/chunk the document
# prepares chunked json objects
# insert the chunked json objects into vector database
@csrf_exempt
@require_http_methods(["POST"])
def upsertContentToVDB_Chunks(request,instructions):
    if request.method == "POST":
        try:
            # Get the query (text input)
            instructions = request.POST.get("query")
            if not instructions:
                return JsonResponse({"error": "Missing query"}, status=400)

            # Get the uploaded file
            if "file" not in request.FILES:
                return JsonResponse({"error": "Missing file"}, status=400)

            # Process file + query
            dataset,file_name = chunk_documents_crud_vdb.upload_and_prepare_file_content_in_chunks(request, instructions)

            if isinstance(dataset, JsonResponse):
                return dataset
            else:
                document_name=file_name+uuid.uuid4().hex
                chunk_documents_crud_vdb.upsertRecord(dataset,document_name=document_name)
                return JsonResponse(
                    {"message": "Data uploaded successfully", "data": dataset},
                    status=200
                )

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


    return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
@require_http_methods(["POST"])
def fetch_records_grouped_by_document_name(request):
    index_n = request.POST.get("index_name")
    records_by_doc=chunk_documents_crud_vdb.fetch_vdb_chunks_grouped_by_document_name(index_n)
    return JsonResponse(records_by_doc, safe=False)



@csrf_exempt
@require_http_methods(["POST"])
def update_records_grouped_by_document_name_in_vdb(request):
    dataset=chunk_documents_crud_vdb.upload_file(request)
    index_n = request.POST.get("index_name")
    transformed_records=chunk_documents_crud_vdb.update_vdb_chunks_grouped_by_document_name(index_n,dataset)
    return JsonResponse(transformed_records, safe=False)

