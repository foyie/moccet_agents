from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
def build_vectorstore(documents, embedding_model):
    return FAISS.from_documents(documents, embedding_model)

def retrieve_combined_context(pdf_vectorstore, json_vectorstore, query, k=2):
    pdf_docs = pdf_vectorstore.similarity_search(query, k=k)
    json_docs = json_vectorstore.similarity_search(query, k=k)
    combined_context = "\n\n".join([doc.page_content for doc in pdf_docs + json_docs])
    return combined_context


def load_healthcare_vectorstore():
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the FAISS vector store from the saved directory
    qa_vectordb = FAISS.load_local(
        "/Users/chandrimadas/Documents/dual_agents/agents_streamlit/vectorstores/healthcare_qa_faiss_db", 
        embedding_model,
        allow_dangerous_deserialization=True  # Add this if using newer versions
    )
    
    return qa_vectordb.as_retriever(search_kwargs={"k": 5})

def load_education_vectorstore():
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the FAISS vector store from the saved directory
    qa_vectordb = FAISS.load_local(
        "/Users/chandrimadas/Documents/dual_agents/agents_streamlit/vectorstores/education_qa_faiss_db", 
        embedding_model,
        allow_dangerous_deserialization=True  # Add this if using newer versions
    )
    
    return qa_vectordb.as_retriever(search_kwargs={"k": 5})

def load_general_vectorstore():
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the FAISS vector store from the saved directory
    qa_vectordb = FAISS.load_local(
        "/Users/chandrimadas/Documents/dual_agents/agents_streamlit/vectorstores/general_qa_faiss_db", 
        embedding_model,
        allow_dangerous_deserialization=True  # Add this if using newer versions
    )
    
    return qa_vectordb.as_retriever(search_kwargs={"k": 5})

