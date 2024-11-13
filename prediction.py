import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
endpoint = os.getenv("ENDPOINT")

# Load the pre-generated embeddings
embeddings_df = pd.read_csv('Customer_Embeddings.csv')

# Check the structure of the Embeddings column
# If the embeddings are stored as strings, ensure they can be properly processed
if isinstance(embeddings_df['Embeddings'][0], str):
    # Convert the embeddings from string representation to np.array
    embeddings_df['Embeddings'] = embeddings_df['Embeddings'].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))

# Define headers for embedding request
headers = {
    "api-key": api_key,
    "Content-Type": "application/json"
}

# Function to generate an embedding for the input demographics
def get_embedding(text):
    data_payload = {
        "input": [text],
        "model": "text-embedding-ada-002"
    }
    response = requests.post(f"{endpoint}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-04-01-preview", headers=headers, json=data_payload)
    
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    else:
        st.error(f"Embedding API Error: {response.status_code} - {response.text}")
        return None

# Function to find the most similar profiles based on cosine similarity
def find_similar_profiles(input_embedding, embeddings_df, top_n=3):
    # Calculate similarity for each stored embedding
    try:
        embeddings_df['similarity'] = embeddings_df['Embeddings'].apply(lambda emb: 1 - cosine(input_embedding, emb))
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return None

    # Sort by similarity and select top N most similar profiles
    return embeddings_df.nlargest(top_n, 'similarity')[['CustomerID', 'similarity']]

# Function to predict using GPT-4o-mini based on similar profiles
def predict_with_gpt4(similar_profiles, task):
    # Format the similar profiles for the GPT-4 prompt
    profiles_text = "\n".join([f"CustomerID: {row['CustomerID']}, Similarity Score: {row['similarity']}" for _, row in similar_profiles.iterrows()])
    prompt = f"Based on the following similar customer profiles:\n{profiles_text}\n  please provide advice on {task}. Do not show the existing customer details"

    # Define the endpoint for the GPT-4o-mini deployment
    if not endpoint or not api_key:
        st.error("Missing API endpoint or API key in environment variables.")
        return None

    url = f"{endpoint}/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"

    # Define headers
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    # Define the request payload
    payload = {
        "messages": [
            {"role": "system", "content": "You are a financial forecasting and risk assistant. please answer concisely and accurately preferred the answer as key value pair formatted (not Json visually). MAke the Key and Values are different colors like green and white"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    # Make the request to the API
    try:
        response = requests.post(url, headers=headers, json=payload)
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"ChatCompletion API Error: {response.status_code} - {response.text}")
        return None

# Streamlit UI
st.title("Customer Demographics Prediction")
st.write("Enter customer demographics to find similar profiles and infer predictions based on similar profiles.")

# Input fields for customer demographics
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Monthly Salary", min_value=0, max_value=100000, value=5000)
job_title = st.selectbox("Job Title", ["Student", "Private Sector Employee", "Government Employee", "Business Owner", "Executive"])

# Format demographics for embedding input
demographics_text = f"Age: {age}, Salary: {salary}, Job Title: {job_title}"

# Button to generate embeddings and predictions
if st.button("Generate Predictions"):
    # Step 1: Generate embedding for input demographics
    input_embedding = get_embedding(demographics_text)

    if input_embedding:
        st.write("Embeddings generated successfully.")

        # Step 2: Find similar profiles
        similar_profiles_df = find_similar_profiles(input_embedding, embeddings_df)

        if similar_profiles_df is not None and not similar_profiles_df.empty:
            st.write("Most similar customer profiles identified:")
            st.dataframe(similar_profiles_df)

            # Step 3: Predictions for each task
            tasks = [
                "the average credit limit we can offer to the new customer as a value between 0 and 100000 ,the credit risk between 0-1000 where 1000 is high risk, you may say moderate, low or high risk ,the expected average credit score between 0 - 1000 ,a social media marketing campaign script to target this customer segment ,                global market opportunity based on these demographics and the current market trends"
            ]

            # Perform inference for each task
            for task in tasks:
                #st.subheader(f"{task.capitalize()}:")
                prediction = predict_with_gpt4(similar_profiles_df, task)
                if prediction:
                    st.write(prediction)
        else:
            st.error("No similar profiles found or similarity calculation failed.")
    else:
        st.error("Failed to generate embedding.")
