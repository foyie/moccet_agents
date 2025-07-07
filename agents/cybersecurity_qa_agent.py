# Cybersecurity Compliance Agent Setup

from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# Cybersecurity QA Prompt
cyber_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a cybersecurity compliance officer and technical specification preparer. Validate business suggestions for NIST, ISO 27001, and SOC 2 compliance, and refine them into technically actionable specifications.\n\n"
     "✅ Instructions:\n"
     "- For each suggestion:\n"
     "  * If fully compliant, refine it into a detailed technical specification.\n"
     "  * If not compliant, modify it to comply.\n"
     "  * Exclude suggestions that cannot be made compliant.\n\n"
     "✅ Output Format:\n"
     "- Provide a numbered list of refined, technically actionable suggestions.\n"
     "- Each must:\n"
     "    • Include security frameworks, encryption methods, and best practices.\n"
     "    • Clearly describe how to ensure NIST, ISO 27001, and SOC 2 compliance.\n"
     "- Do not include introductions or explanations. Only output the refined suggestions.\n"
    ),
    ("user", "{context}\n\nSuggestions to validate:\n{suggestions}")
])

cyber_qa_chain = LLMChain(llm=llm, prompt=cyber_qa_prompt)

def run_cybersecurity_qa(suggestions, retriever):
    query = "Does this comply with NIST, ISO 27001, and SOC 2 regulations?"
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    cyber_response = cyber_qa_chain.invoke({"context": context, "suggestions": suggestions})
    return cyber_response["text"].strip()

# Cybersecurity Compliance Sources
cyber_urls = [
    "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
    "https://www.iso.org/isoiec-27001-information-security.html",
    "https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/socforserviceorganizations.html"
]

# Load web pages
cyber_loaders = [WebBaseLoader(url) for url in cyber_urls]
cyber_docs = []
for loader in cyber_loaders:
    cyber_docs.extend(loader.load())

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
cyber_chunks = splitter.split_documents(cyber_docs)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build Cybersecurity Vector DB
cyber_vectordb = Chroma.from_documents(cyber_chunks, embedding=embedding_model, persist_directory="./cyber_qa_chroma_db")
# Build Retriever
cyber_qa_retriever = cyber_vectordb.as_retriever(search_kwargs={"k": 5})
