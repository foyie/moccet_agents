import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_json_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    documents = []
    for entry in data:
        documents.append(Document(page_content=f"Brief: {entry['brief']}\nSuggestions: {entry['suggestions']}"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    return chunks
