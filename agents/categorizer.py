from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
# chain = LLMChain(llm=llm, prompt=prompt)

# def load_categorizer(llm):
#     categorization_prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a classification engine. Categorize the entire set of business suggestions into one of the following domains:\n\n"
#          "- Healthcare\n- Education\n- General\n\n"
#          "✅ Instructions:\n"
#          "Analyze the suggestions and return only one category name based on the primary focus.\n\n"
#          "✅ Output Format:\n"
#          "Healthcare\nOR\nEducation\nOR\nGeneral"),
#         ("user", "Business Suggestions:\n{suggestions}")
#     ])
#     return LLMChain(llm=llm, prompt=categorization_prompt)

# def categorize_suggestions(chain, suggestions):
#     result = chain.invoke({"suggestions": suggestions})
#     return result['text'].strip()



from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Categorization prompt for the entire suggestion block
categorization_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a classification engine. Categorize the entire set of business suggestions into one of the following domains:\n\n"
     "- Healthcare\n- Education\n- General\n- Cybersecurity\n- Finance\n\n"
     "✅ Instructions:\n"
     "Analyze the suggestions and return only one category name based on the primary focus.\n\n"
     "✅ Output Format:\n"
     "Healthcare\nOR\nEducation\nOR\nFinance\nOR\nCybersecurity\nOR\nGeneral"),
    ("user", "Business Suggestions:\n{suggestions}")
])

categorization_chain = LLMChain(llm=llm, prompt=categorization_prompt)

def categorize_suggestions(suggestions):
    result = categorization_chain.invoke({"suggestions": suggestions})
    return result['text'].strip()

