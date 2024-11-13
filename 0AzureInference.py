import os
import requests
import pandas as pd
import time
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import AzureSearch

# Load environment variables
load_dotenv(override=True)

# Set up Azure Cognitive Search and OpenAI details
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
ENDPOINT = os.getenv("ENDPOINT")
API_KEY = os.getenv("API_KEY")

# Set up LangChain embedding and Azure Search connection
embedding_model_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=API_KEY,
    openai_api_base=f"{ENDPOINT}"
)

# Configure Azure Search vector store
azure_search = AzureSearch(
    azure_search_service=AZURE_SEARCH_SERVICE,
    index_name=AZURE_SEARCH_INDEX,
    azure_search_endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
    azure_search_key=AZURE_SEARCH_API_KEY,
    embedding_function=embeddings.embed_query
)

# Define prompt template for GPT-4
prompt_template = PromptTemplate(
    input_variables=["customer_details", "similar_profiles"],
    template="""
    Based on the customer characteristics: {customer_details}, and taking into account the following similar customer profiles: {similar_profiles}, 
    provide a credit score, credit risk level, and suggested mitigation techniques.
    """
)

llm = OpenAI(
    model_name="gpt-4",
    openai_api_key=API_KEY,
    openai_api_base=f"{ENDPOINT}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to create embeddings and search similar profiles
def search_similar_profiles(customer_description):
    # Generate embedding for the customer description
    embedding = embeddings.embed_query(customer_description)
    # Search for similar profiles
    similar_profiles = azure_search.similarity_search(embedding, k=5)
    return similar_profiles

# Function to get inference from GPT-4 model
def get_gpt4_inference(customer_details, similar_profiles):
    similar_profiles_summary = "\n".join([profile["content"] for profile in similar_profiles])
    result = llm_chain.run({
        "customer_details": customer_details,
        "similar_profiles": similar_profiles_summary
    })
    return result

# Streamlit UI setup
import streamlit as st

st.title("Customer Insights and Analysis with LangChain")

# User inputs for analysis
gender = st.selectbox("Gender", ("Male", "Female", "Other"), index=0)
age = st.number_input("Age", min_value=0, step=1, value=55)
marital_status = st.selectbox("Marital Status", ("Single", "Married", "Divorced", "Widowed"), index=1)
occupation = st.text_input("Occupation", value="Teacher")
income_level = st.text_input("Income Level", value="12500")
education_level = st.text_input("Education Level")

# Button to create credit score and risk assessment
if st.button("Create Credit Score and Risk Assessment"):
    try:
        customer_description = (
            f"Gender: {gender}, Age: {age}, Marital Status: {marital_status}, "
            f"Occupation: {occupation}, Income Level: {income_level}, Education Level: {education_level}"
        )
        similar_profiles = search_similar_profiles(customer_description)
        if similar_profiles:
            gpt4_response = get_gpt4_inference(customer_description, similar_profiles)
            st.header("Credit Score and Risk Assessment")
            st.write(gpt4_response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
