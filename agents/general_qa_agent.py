from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import LLMChain
# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your LLM (replace with your preferred model)
from langchain_google_genai import ChatGoogleGenerativeAI

general_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a compliance officer for general business regulations. Validate business suggestions for GDPR, CCPA, and FTC compliance, and refine them into technically actionable specifications.\n\n"
     "✅ Instructions:\n"
     "- For each suggestion:\n"
     "  * If fully compliant, refine it into a detailed technical specification.\n"
     "  * If not compliant, modify it to comply.\n"
     "  * Exclude suggestions that cannot be made compliant.\n\n"
     "✅ Output Format:\n"
     "- Provide a numbered list of refined, technically actionable suggestions.\n"
     "- Each must:\n"
     "    • Include technologies, frameworks, and implementation methods.\n"
     "    • Clearly describe how to ensure GDPR, CCPA, and FTC compliance.\n"
     "- Do not include introductions or explanations. Only output the refined suggestions.\n"
    ),
    ("user", "{context}\n\nSuggestions to validate:\n{suggestions}")
])
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
general_qa_chain = LLMChain(llm=llm, prompt=general_qa_prompt)

def run_general_qa(suggestions, retriever):
    query = "Does this comply with GDPR, CCPA, and FTC regulations?"
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    general_response = general_qa_chain.invoke({"context": context, "suggestions": suggestions})
    return general_response["text"].strip()