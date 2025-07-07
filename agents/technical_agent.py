# from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate

# def build_technical_agent(llm):
#     tech_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a technical solutions architect. Based on the provided business suggestions and technical documentation, provide detailed, step-by-step implementation strategies, tools, and best practices."),
#         ("user", "{tech_context}\n\nBusiness Suggestions: {business_suggestions}")
#     ])
#     return LLMChain(llm=llm, prompt=tech_prompt)

# def retrieve_tech_context(vectorstore, business_suggestions, k=3):
#     docs = vectorstore.similarity_search(business_suggestions, k=k)
#     return "\n\n".join([doc.page_content for doc in docs])



# from langchain.vectorstores import Chroma
# # from langchain.embeddings import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain

# # Load your tech vector store
# tech_vectordb = Chroma(persist_directory="/Users/chandrimadas/Documents/dual_agents/full_tech_vectorstore", embedding_function=embedding_model)

# # Initialize Gemini model
# chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# # # Define technical agent prompt
# # tech_prompt = ChatPromptTemplate.from_messages([
# #     ("system", "You are a technical solutions architect. Based on the provided business suggestions and technical documentation, provide detailed, step-by-step implementation strategies, tools, and best practices. Be specific and include architecture, services, and data considerations."),
# #     ("user", "{tech_context}\n\nBusiness Suggestions: {business_suggestions}")
# # ])

# # # Set up the LLM chain for the technical agent
# # tech_llm_chain = LLMChain(llm=chat_model, prompt=tech_prompt)


# # Function to retrieve technical context
# # def retrieve_tech_context(vectorstore, business_suggestions, k=3):
# #     docs = vectorstore.similarity_search(business_suggestions, k=k)
# #     return "\n\n".join([doc.page_content for doc in docs])


# # Complete technical agent pipeline
# def technical_agent_pipeline(business_suggestions):
#     # Retrieve the most relevant technical docs
#     tech_context = retrieve_tech_context(tech_vectordb, business_suggestions, k=3)
    
#     # Get implementation strategies from Gemini
#     response = tech_llm_chain.run(tech_context=tech_context, business_suggestions=business_suggestions)
    
#     return response



from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
#     # Load the FAISS vector store from the saved directory
# qa_vectordb = FAISS.load_local(
#         "/Users/chandrimadas/Documents/agents_streamlit/vectorstores/healthcare_qa_faiss_db", 
#         embedding_model,
#         allow_dangerous_deserialization=True  # Add this if using newer versions
#     )
    
#     return qa_vectordb.as_retriever(search_kwargs={"k": 5})

tech_vectordb = FAISS.load_local(
        "/Users/chandrimadas/Documents/agents_streamlit/vectorstores/tech_faiss_db", 
        embedding_model,
        allow_dangerous_deserialization=True  # Add this if using newer versions
    )
# Chroma(persist_directory="./data/full_tech_vectorstore")

chat_model = llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
# ChatGoogleGenerativeAI(model="gemini-1.5-flash")

tech_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical solutions architect. Based on the provided business suggestions and technical documentation, provide detailed, step-by-step implementation strategies, tools, and best practices. Be specific and include architecture, services, and data considerations."),
    ("user", "{tech_context}\n\nBusiness Suggestions: {business_suggestions}")
])

tech_llm_chain = LLMChain(llm=chat_model, prompt=tech_prompt)

def retrieve_tech_context(vectorstore, business_suggestions, k=3):
    docs = vectorstore.similarity_search(business_suggestions, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def technical_agent_pipeline(business_suggestions):
    tech_context = retrieve_tech_context(tech_vectordb, business_suggestions, k=3)
    response = tech_llm_chain.run(tech_context=tech_context, business_suggestions=business_suggestions)
    return response
