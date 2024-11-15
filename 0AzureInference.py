import os
import requests
import json
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
from langchain_openai import ChatOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import streamlit as st

# Load environment variables
load_dotenv(override=True)

# Set up Azure Cognitive Search and OpenAI details
openai.api_type ="azure"
AZURE_SEARCH_SERVICE = st.secrets["AZURE_SEARCH_SERVICE"]
AZURE_SEARCH_INDEX = st.secrets["AZURE_SEARCH_INDEX"]
AZURE_SEARCH_API_KEY = st.secrets["AZURE_SEARCH_API_KEY"]
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = st.secrets["OPENAI_API_VERSION"]
api_key = st.secrets["AZURE_INFERENCE_CREDENTIAL"]

# Set up LangChain embedding and Azure Search connection using AzureOpenAIEmbeddings
print("[DEBUG] Setting up AzureOpenAIEmbeddings...")
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_openai_api_key=AZURE_OPENAI_API_KEY,
    azure_openai_api_version=AZURE_OPENAI_API_VERSION
)
print("[DEBUG] AzureOpenAIEmbeddings setup complete.")

# Define the Azure Cognitive Search endpoint
api_version = "2021-04-30-Preview"
search_url = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={api_version}"
count_url = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/$count?api-version={api_version}"
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_API_KEY
}



# Configure Azure AI Search retriever
print("[DEBUG] Configuring Azure AI Search retriever...")
retriever = AzureAISearchRetriever(
    service_name=AZURE_SEARCH_SERVICE,
    index_name=AZURE_SEARCH_INDEX,
    api_key=AZURE_SEARCH_API_KEY,
    content_key="content",
    top_k=5
)
print("[DEBUG] Azure AI Search retriever configuration complete.")

# Set up ChatOpenAI LLM
print("[DEBUG] Setting up ChatOpenAI LLM...")

llm = ChatOpenAI(
    api_key="inin3F2ATscXKBSrFiXrqS1kJMQhoITEAV",
    base_url="https://Phi-3-5-mini-instruct-rivhp.southcentralus.models.ai.azure.com/v1/chat/completions",
    model="phi35-mini-instruct"
)
print("[DEBUG] ChatOpenAI LLM setup complete.")

# Function to create embeddings and search similar profiles
def search_similar_profiles(customer_description):
    print(f"[DEBUG] Searching for similar profiles with description: {customer_description}")
    similar_profiles = retriever.invoke(customer_description)
    print(f"[DEBUG] Retrieved similar profiles: {similar_profiles}")
    return similar_profiles

# Streamlit UI setup
import streamlit as st

st.title("Customer Insights and Analysis with Enadoc Azure AI")

# Display number of vectors in the store
try:
    count_response = requests.get(count_url, headers=headers)
    if count_response.status_code == 200:
        vector_count = count_response.text
        st.write(f"Total number of vectors in the store: {vector_count}")
    else:
        st.error(f"Failed to retrieve vector count: {count_response.status_code} - {count_response.text}")
except Exception as e:
    st.error(f"An error occurred while retrieving vector count: {str(e)}")

# User inputs for analysis
gender = st.selectbox("Gender", ("Male", "Female", "Other"), index=0)
age = st.number_input("Age", min_value=0, step=1, value=55)
marital_status = st.selectbox("Marital Status", ("Single", "Married", "Divorced", "Widowed"), index=1)
occupation = st.text_input("Occupation", value="Teacher")
income_level = st.text_input("Income Level", value="12500")
education_level = st.text_input("Education Level")

# Button to search similar profiles
if st.button("Search Similar Profiles"):
    try:
        customer_info = {
            "Gender": gender,
            "Age": age,
            "Marital Status": marital_status,
            "Occupation": occupation,
            "Income Level": income_level,
            "Education Level": education_level
        }
        print(f"[DEBUG] Customer info: {customer_info}")
        if (similar := search_similar_profiles(json.dumps(customer_info))):
            st.session_state.similar_profiles = similar
            st.header("Similar Customer Profiles")
            for profile in similar:
                st.write(profile.page_content)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    print(f"[ERROR] {str(e)}")

# Button to create credit score and risk assessment
if st.button("Create Credit Score and Risk Assessment"):
    try:
        if 'similar_profiles' in st.session_state and st.session_state.similar_profiles:
            customer_info = {
                "Gender": gender,
                "Age": age,
                "Marital Status": marital_status,
                "Occupation": occupation,
                "Income Level": income_level,
                "Education Level": education_level
            }
            print(f"[DEBUG] Customer info: {customer_info}")
            new_customer= json.dumps(customer_info).replace("{", "").replace("}", "").replace('"', '')
            print(f"[DEBUG] Customer string: {new_customer}")
            raw_data = "'" "".join([profile.page_content for profile in st.session_state.similar_profiles])+"'"
            print(f"[DEBUG] Combined description: {raw_data}")
            #gpt4_response = llm.invoke("tell me a story")
            #gpt4_response = client.completions.create(model=deployment_name, prompt="tell me a story", max_tokens=100)
            st.header("Credit Score and Risk Assessment")
            
        # GPT-4 Prompt
        prompt = f"""
        You are a financial expert tasked with analyzing customer profiles to predict credit scores, risk levels, and mitigation strategies.

        The dataset below contains details of similar customers, including their credit scores and financial metrics:
        {raw_data}

        A new customer has the following demographic details:
        {new_customer}

        Based on the patterns in the dataset, predict the following for the new customer:
        1. Estimated Credit Score
        2. Risk Profile (e.g., High Risk, Medium Risk, Low Risk)
        3. Recommended Risk Mitigation Strategies

        Provide a detailed explanation for your predictions.
        """

        # Call GPT-4 for inference
        if raw_data:
            
            client = AzureOpenAI(
            azure_endpoint = AZURE_OPENAI_ENDPOINT, 
            api_key=AZURE_OPENAI_API_KEY,  
            api_version=OPENAI_API_VERSION
            )

            response = client.chat.completions.create(
                model="gpt-4", # model = "deployment_name"
                messages=[
                    {"role": "system", "content": "You are a financial expert."},
                    {"role": "user", "content": prompt},
                ]
            )           
            # Output GPT-4's response\
            print("GPT-4 Analysis and Predictions:")
            print(response.choices[0].message.content)
            st.write(response.choices[0].message.content)
        else:
            st.error("Please search for similar profiles first.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

