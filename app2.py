import os
import sys
import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the Gemini model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# # Set the working directory to the project root
# project_root = os.path.dirname(os.path.abspath(__file__))
# os.chdir('/Users/chandrimadas/Documents/agents_streamlit')

# Optional: Add project root to Python path
sys.path.append('/Users/chandrimadas/Documents/dual_agents/agents_streamlit')

import streamlit as st
from agents.business_agent import business_agent_pipeline
from agents.health_qa_agent import run_healthcare_qa
from agents.education_qa_agent import run_education_qa
from agents.general_qa_agent import run_general_qa
from agents.categorizer import categorize_suggestions
from agents.technical_agent import technical_agent_pipeline
# from agents.technical_agent import build_technical_agent, retrieve_tech_context

from utils.file_processing import save_uploaded_file
# from utils.vectorstore_retriever import load_healthcare_vectorstore

from utils.vectorstore_retriever import load_healthcare_vectorstore, load_education_vectorstore, load_general_vectorstore

st.set_page_config(page_title="Dual Agents Business Optimizer", layout="wide")
st.title("📊 Multiple Agents Business Optimizer")

# Initialize healthcare retriever
health_qa_retriever = load_healthcare_vectorstore()
education_qa_retriever = load_education_vectorstore()
general_qa_retriever = load_general_vectorstore()

# Step 1: Upload file and input query
uploaded_file = st.file_uploader("Upload a Business Brief (PDF)", type="pdf")
user_question = st.text_input("Enter Your Business Question")

if uploaded_file and user_question:
    with st.spinner("Processing business suggestions..."):
        file_path = save_uploaded_file(uploaded_file)
        suggestions = business_agent_pipeline(file_path, user_question, "/Users/chandrimadas/Documents/dual_agents/agents_streamlit/business_agent_dataset_detailed.json")
    
    st.subheader("🔍 Business Agent Suggestions")
    st.write(suggestions)

    with st.spinner("Validating suggestions..."):
        category = categorize_suggestions(suggestions)
        # validated_suggestions = run_education_qa(suggestions, education_qa_retriever)
        if category == "Education":
            
            validated_suggestions = run_education_qa(suggestions, education_qa_retriever)
            st.error(f"Currently, this system only supports Healthcare QA. Detected category: {category}")
        if category=="Healthcare":
            validated_suggestions = run_healthcare_qa(suggestions, health_qa_retriever)
        if category=="Finance":
            validated_suggestions = run_healthcare_qa(suggestions, health_qa_retriever)
        if category=="Cybersecurity":
            validated_suggestions = run_healthcare_qa(suggestions, health_qa_retriever)                    
            
            
        else:
            validated_suggestions = run_general_qa(suggestions, general_qa_retriever)
            
        # if category != "Healthcare":
        #     st.error(f"Currently, this system only supports Healthcare QA. Detected category: {category}")
        # else:
        #     validated_suggestions = run_healthcare_qa(suggestions, health_qa_retriever)

        st.subheader("✅ Healthcare QA Validated Suggestions")
        st.write(validated_suggestions)
            # options = validated_suggestions.split("\n")
            # selected_suggestion = st.selectbox("Select a Suggestion for Technical Planning", options)
            

# st.subheader("✅ Healthcare QA Validated Suggestions")
# st.write(validated_suggestions)

# Split the validated_suggestions into numbered options
# The pattern splits on the numbers followed by a period and a space (e.g., '1. ')
        # options = re.split(r'\n?\d+\.\s', validated_suggestions)

            # Remove any empty strings and the text before the first bullet point
        # options = [option.strip() for option in options if option.strip()]

            # Multi-select box
        # selected_suggestion = st.multiselect("Select Suggestions for Technical Planning", options)

            # # Optional: Display selected suggestions
            # if selected_suggestions:
            #     st.subheader("🛠️ Selected Suggestions for Planning")
            
            #     for suggestion in selected_suggestions:
            #         st.write(suggestion)
        
        selected_suggestion = st.text_area("Select and refine the suggestion you want to implement:", validated_suggestions)
        # Multi-select box
        # selected_suggestions = st.multiselect("Select Suggestions for Technical Planning", options)

        # Button to generate technical plan
        # if st.button("Generate Technical Plan"):
        #     if selected_suggestions:
        #         combined_suggestions = " ".join(selected_suggestions)  # 🔥 FIX: Join list into a string
        #         with st.spinner("Generating technical implementation plan..."):
        #             technical_plan = technical_agent_pipeline(combined_suggestions)
        #         st.subheader("🛠️ Technical Implementation Plan")
        #         st.write(technical_plan)
        #     else:
        #         st.warning("Please select at least one suggestion.")


        if st.button("Generate Technical Plan"):
            with st.spinner("Generating technical implementation plan..."):
                technical_plan = technical_agent_pipeline(selected_suggestion)
            st.subheader("🛠️ Technical Implementation Plan")
            st.write(technical_plan)
                


        # st.subheader("✅ QA Validated Suggestions")



    # else:
    #     st.warning("Please upload a PDF and enter a question.")
        
