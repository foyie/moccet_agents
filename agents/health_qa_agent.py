from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a healthcare compliance officer and technical specification preparer. Your task is to validate business suggestions for compliance with HIPAA, FDA, and HHS regulations and refine them into technically actionable specifications.\n\n"
     "✅ Instructions:\n"
     "- For each suggestion:\n"
     "  * If fully compliant, refine it into a detailed technical specification.\n"
     "  * If not compliant, modify it to comply.\n"
     "  * Exclude suggestions that cannot be made compliant.\n\n"
     "✅ Output Format:\n"
     "- Provide a numbered list of refined, technically actionable suggestions.\n"
     "- Each suggestion must:\n"
     "    • Include specific technologies, frameworks, or implementation methods.\n"
     "    • Clearly describe how to ensure regulatory compliance (e.g., encryption standard, API security protocol).\n"
     "- Do not include introductions or explanations. Only output the refined suggestions.\n"
    ),
    ("user", "{context}\n\nSuggestions to validate:\n{suggestions}")
])

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your LLM (replace with your preferred model)
from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

def run_healthcare_qa(suggestions, retriever):
    query = "Does this comply with HIPAA, FDA, and HHS guidelines?"
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    qa_response = qa_chain.invoke({"context": context, "suggestions": suggestions})
    return qa_response["text"].strip()
