import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
from langchain_openai import ChatOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv(override=True)

# Set up Azure Cognitive Search and OpenAI details
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL")

# Set up LangChain embedding and Azure Search connection using AzureOpenAIEmbeddings
print("[DEBUG] Setting up AzureOpenAIEmbeddings...")
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_openai_api_key=AZURE_OPENAI_API_KEY,
    azure_openai_api_version=AZURE_OPENAI_API_VERSION
)
print("[DEBUG] AzureOpenAIEmbeddings setup complete.")

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

st.title("Customer Insights and Analysis with LangChain")

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
                endpoint='https://Phi-3-5-vision-instruct-vmdvw.southcentralus.models.ai.azure.com',
                credential=AzureKeyCredential(api_key))

            model_info = client.get_model_info()
            print("Model name:", model_info.model_name)
            print("Model type:", model_info.model_type)
            print("Model provider name:", model_info.model_provider_name)

            payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "please analyse the following data and provide and generate a credit score and risk assessment, first row represent new customer and the rest are existing customers with more data such as late payment, credit score type of card etc., we need to generate the credit risk assesment for the first (new) customer"
                },
                {
                "role": "user",
                "content": "{combined_description}"
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.8,
            "top_p": 0.1,
            "presence_penalty": 0,
            "frequency_penalty": 0
            }
            response = client.complete(payload).choices[0].message.content
            st.write(response)
        else:
            st.error("Please search for similar profiles first.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"[ERROR] {str(e)}")
