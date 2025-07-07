from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

finance_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial compliance officer and technical specification preparer. Validate business suggestions for PCI-DSS, SOX, and SEC reporting compliance, and refine them into technically actionable specifications.\n\n"
     "✅ Instructions:\n"
     "- For each suggestion:\n"
     "  * If fully compliant, refine it into a detailed technical specification.\n"
     "  * If not compliant, modify it to comply.\n"
     "  * Exclude suggestions that cannot be made compliant.\n\n"
     "✅ Output Format:\n"
     "- Provide a numbered list of refined, technically actionable suggestions.\n"
     "- Each must:\n"
     "    • Include encryption, transaction logging, and financial reporting standards.\n"
     "    • Clearly describe how to ensure PCI-DSS, SOX, and SEC compliance.\n"
     "- Do not include introductions or explanations. Only output the refined suggestions.\n"
    ),
    ("user", "{context}\n\nSuggestions to validate:\n{suggestions}")
])

finance_qa_chain = LLMChain(llm=llm, prompt=finance_qa_prompt)

def run_finance_qa(suggestions, retriever):
    query = "Does this comply with PCI-DSS, SOX, and SEC reporting regulations?"
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    finance_response = finance_qa_chain.invoke({"context": context, "suggestions": suggestions})
    return finance_response["text"].strip()