import json
import warnings  # Correct module for warnings, including PendingDeprecationWarning
import os
from docx import Document
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import uuid
from openai import OpenAI
from openai import OpenAI
from .llm_wrapper import LLMClientWrapper  # Relative import

import tempfile
# Initialize OpenAI client
client =  OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from django.core.files.uploadedfile import UploadedFile

def extract_content(file):
    """
    Extract text content from a file.
    Supports:
    - PDF (.pdf)
    - Word (.docx)
    - Text (.txt)
    - Images (.png, .jpg, .jpeg, .tiff, .bmp)

    file: either a file path (str) or a Django UploadedFile object (request.FILES['file'])
    """
    # Determine if input is file path or UploadedFile
    if isinstance(file, UploadedFile):
        filename = file.name
        ext = os.path.splitext(filename)[1].lower()

        # Check if file is already on disk
        if hasattr(file, 'temporary_file_path'):
            filepath = file.temporary_file_path()
        else:
            # Save in-memory file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in file.chunks():
                    tmp.write(chunk)
                filepath = tmp.name
    else:
        # It's a file path
        filepath = file
        ext = os.path.splitext(filepath)[1].lower()

    text = ""

    if ext == ".pdf":
        reader = PdfReader(filepath)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])

    elif ext == ".docx":
        doc = Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])

    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".json":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            # Convert JSON to text (pretty print or flatten)
            text = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # If we created a temporary file, optionally delete it
    if isinstance(file, UploadedFile) and not hasattr(file, 'temporary_file_path'):
        try:
            os.remove(filepath)
        except Exception:
            pass

    return text.strip()




def split_text_by_words(text, chunk_size=200):
    """Split text into chunks of N words."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_with_llm(chunk,instructions):
    """
    Send a chunk to LLM and return structured JSON array.
    Expected format: [{"id": ..., "title": ..., "text": ...}, ...]
    """
    context = f"""
    Take the following text and split it into sections based on the most important category headings (ignore lower level headings).
    For each section, return a JSON object with - no extra words other than the json and remove ```json:
    - "id" (a UUID you generate),
    - "title" (the most important heading),
    - "text" (the remaining text under that heading).

    Text:
    {chunk}
    """
    instructions=instructions or "Exract sections"
    system_prompt="You are a helpful assistant that structures text into JSON sections."
    # Create an instance of your LLM wrapper
    llm = LLMClientWrapper(client, model="gpt-4o-mini", temperature=0, system_prompt=system_prompt)
    response=llm.chat(context,instructions)

    # Parse JSON safely
    try:
        structured_data = eval(response)
    except Exception:
        structured_data = []

    return structured_data


def process_large_text(text, instructions,chunk_size=200):
    """Main function: split -> send to LLM -> collect results."""
    chunks = split_text_by_words(text, chunk_size)
    all_results = []

    for chunk in chunks:
        structured = process_with_llm(chunk,instructions)
        # Ensure UUIDs exist
        for obj in structured:
            if "id" not in obj:
                obj["id"] = str(uuid.uuid4())
        all_results.extend(structured)

    return all_results



def prepare_chunked_text(uploaded_file,instructions):
    content =extract_content(uploaded_file)
    results=process_large_text(content,instructions, chunk_size=200)
    print (results)
    return results






















































