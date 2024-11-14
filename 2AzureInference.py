import os
import requests
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
from langchain_openai import ChatOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import streamlit as st

# Load environment variables
#load_dotenv(override=True)

# Set up Azure Cognitive Search and OpenAI details
AZURE_SEARCH_SERVICE = st.secrets["AZURE_SEARCH_SERVICE"]
AZURE_SEARCH_INDEX = st.secrets["AZURE_SEARCH_INDEX"]
AZURE_SEARCH_API_KEY = st.secrets["AZURE_SEARCH_API_KEY"]
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
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
        similar_profiles = search_similar_profiles(json.dumps(customer_info))
        if similar_profiles:
            st.session_state.similar_profiles = similar_profiles
            st.header("Similar Customer Profiles")
            for profile in similar_profiles:
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
            customer_str= json.dumps(customer_info).replace("{", "").replace("}", "").replace('"', '')
            print(f"[DEBUG] Customer string: {customer_str}")
            combined_description = "'"+customer_str + "".join([profile.page_content for profile in st.session_state.similar_profiles])+"'"
            print(f"[DEBUG] Combined description: {combined_description}")
            #gpt4_response = llm.invoke("tell me a story")
            #gpt4_response = client.completions.create(model=deployment_name, prompt="tell me a story", max_tokens=100)
            st.header("Credit Score and Risk Assessment")
            
            client = ChatCompletionsClient(
                endpoint='https://Phi-3-5-vision-instruct-aohdz.southcentralus.models.ai.azure.com',
                credential=AzureKeyCredential(api_key))

            model_info = client.get_model_info()
            print("Model name:", model_info.model_name)
            print("Model type:", model_info.model_type)
            print("Model provider name:", model_info.model_provider_name)

            payload = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "You are an AI assistant that helps create credit scores and risk analysis for new customers requesting credit cards. You will always be provided with a dataset where the first record will contain the new customer information and 5 similar customers from our database. Based on their performance, you need to predict the credit scores and risk profiles."
                    },
                    {
                        "role": "user",
                        "content": "Analyze this: {combined_description} \nPlease give me prediction as key-value pairs for Credit Score, Credit Risk, and Comments."
                    }
                ],
            "max_tokens": 2048,
            "temperature": 0.8,
            "top_p": 0.1,
            "presence_penalty": 0,
            "frequency_penalty": 0
            }
            response = client.complete(payload).choices[0].message.content
            print("response:",payload)
            st.write(response)
        else:
            st.error("Please search for similar profiles first.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"[ERROR] {str(e)}")
