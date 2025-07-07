import json
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assume llm and embedding_model will be provided from app.py

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    return chunks

def process_json_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    documents = []
    for entry in data:
        documents.append(Document(page_content=f"Brief: {entry['brief']}\nSuggestions: {entry['suggestions']}"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    return chunks

def build_vectorstore(documents, embedding_model):
    return FAISS.from_documents(documents, embedding_model)

def retrieve_combined_context(pdf_vectorstore, json_vectorstore, query, k=2):
    pdf_docs = pdf_vectorstore.similarity_search(query, k=k)
    json_docs = json_vectorstore.similarity_search(query, k=k)

    combined_context = "\n\n".join([doc.page_content for doc in pdf_docs + json_docs])
    return combined_context

def business_agent_pipeline(client_pdf_path, client_query, json_dataset_path, llm, embedding_model):
    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business consultant. Analyze the provided documents and generate 3 actionable, section-referenced suggestions to improve the business model. Focus on revenue, operational efficiency, and customer satisfaction."),
        ("user", "{context}\n\n{question}")
    ])
    chain = LLMChain(llm=llm, prompt=prompt)

    # Process client PDF
    pdf_chunks = load_and_split_pdf(client_pdf_path)
    pdf_vectorstore = build_vectorstore(pdf_chunks, embedding_model)

    # Process JSON dataset
    json_chunks = process_json_dataset(json_dataset_path)
    json_vectorstore = build_vectorstore(json_chunks, embedding_model)

    # Retrieve context
    context = retrieve_combined_context(pdf_vectorstore, json_vectorstore, client_query, k=2)

    # Run the LLM chain
    response = chain.invoke({"context": context, "question": client_query})
    return response["text"]
