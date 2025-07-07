
# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# import os
# import json
# from dotenv import load_dotenv
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# from langchain.docstore.document import Document
# from langchain.document_loaders import PyPDFLoader
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Initialize the embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Initialize the Gemini model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
# urls = [
#     "https://www.hhs.gov/hipaa/for-professionals/privacy/index.html",
#     "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/mobile-medical-applications",
#     "https://www.hhs.gov/ash/patient-safety/index.html"
# ]

# # Load web pages directly
# loaders = [WebBaseLoader(url) for url in urls]
# docs = []
# for loader in loaders:
#     docs.extend(loader.load())


# splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
# chunks = splitter.split_documents(docs)

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# qa_vectordb = FAISS.from_documents(chunks, embedding=embedding_model)

# # Save the vector store
# qa_vectordb.save_local("/Users/chandrimadas/Documents/agents_streamlit/vectorstores/health_qa_faiss_db")
# # qa_vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="./health_qa_chroma_db")

# # qa_retriever = qa_vectordb.as_retriever(search_kwargs={"k": 5})

# from langchain.document_loaders import WebBaseLoader
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings

# # URLs for Education Compliance
# edu_urls = [
#     "https://studentprivacy.ed.gov/",
#     "https://www.ftc.gov/business-guidance/privacy-security/childrens-privacy",
#     "https://www.ada.gov/resources/overview/"
# ]

# # Load web pages
# edu_loaders = [WebBaseLoader(url) for url in edu_urls]
# edu_docs = []
# for loader in edu_loaders:
#     edu_docs.extend(loader.load())

# # Split documents
# splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
# edu_chunks = splitter.split_documents(edu_docs)

# # Initialize embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# # Build Education Vector DB
# edu_qa_vectordb = FAISS.from_documents(edu_chunks, embedding=embedding_model)
# edu_qa_vectordb.save_local("/Users/chandrimadas/Documents/agents_streamlit/vectorstores/edu_qa_faiss_db")
# # Build Retriever
# # edu_qa_retriever = edu_qa_vectordb.as_retriever(search_kwargs={"k": 5})

# general_urls = [
#     "https://gdpr-info.eu/",
#     "https://oag.ca.gov/privacy/ccpa",
#     "https://www.ftc.gov/business-guidance/privacy-security"
# ]

# # Load web pages
# general_loaders = [WebBaseLoader(url) for url in general_urls]
# general_docs = []
# for loader in general_loaders:
#     general_docs.extend(loader.load())

# # Split documents
# general_chunks = splitter.split_documents(general_docs)

# # Build General Vector DB
# general_qa_vectordb = FAISS.from_documents(general_chunks, embedding=embedding_model)
# general_qa_vectordb.save_local("/Users/chandrimadas/Documents/agents_streamlit/vectorstores/general_qa_faiss_db")
import os
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize embedding model and LLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# Common text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)


def build_vector_db(urls, db_path):
    """
    Load documents from URLs, split them, build FAISS vector store, and save locally.
    """
    print(f"Processing URLs for: {db_path}")

    # Load web pages
    loaders = [WebBaseLoader(url) for url in urls]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Split documents
    chunks = splitter.split_documents(docs)

    # Build and save FAISS vector store
    vectordb = FAISS.from_documents(chunks, embedding=embedding_model)
    vectordb.save_local(db_path)
    print(f"Saved vector database to: {db_path}\n")

    return vectordb


# Define industries and their sources
industry_sources = {
    "healthcare": [
        "https://www.hhs.gov/hipaa/for-professionals/privacy/index.html",
        "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/mobile-medical-applications",
        "https://www.hhs.gov/ash/patient-safety/index.html"
    ],
    "education": [
        "https://studentprivacy.ed.gov/",
        "https://www.ftc.gov/business-guidance/privacy-security/childrens-privacy",
        "https://www.ada.gov/resources/overview/"
    ],
    "general": [
        "https://gdpr-info.eu/",
        "https://oag.ca.gov/privacy/ccpa",
        "https://www.ftc.gov/business-guidance/privacy-security"
    ],
    "cybersecurity":  [
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "https://www.iso.org/isoiec-27001-information-security.html",
        "https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/socforserviceorganizations.html"
    ],
    "finance":[
    "https://www.pcisecuritystandards.org/document_library",
    "https://www.sec.gov/rules/final.shtml",
    "https://pcaobus.org/oversight/standards/archived-standards/sarbanes-oxley-act-of-2002"
]
    
}

# Build vector databases for each industry
vector_db_paths = {
    industry: f"/Users/chandrimadas/Documents/dual_agents/agents_streamlit/vectorstores/{industry}_qa_faiss_db"
    for industry in industry_sources
}

for industry, urls in industry_sources.items():
    build_vector_db(urls, vector_db_paths[industry])

