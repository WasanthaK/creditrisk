import streamlit as st
import requests
import os
import numpy as np
from chromadb import Client
from dotenv import load_dotenv
from chromadb.config import Settings

# Load environment variables
load_dotenv()
api_key = os.getenv("EMBEDDING_KEY")
endpoint = os.getenv("EMBEDDING_END_POINT")

# Define the URL for the embeddings endpoint
url = f"{endpoint}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-04-01-preview"

# Define the headers
headers = {
    "api-key": api_key,
    "Content-Type": "application/json"
}


import pandas as pd
import time

# Initialize ChromaDB client

chroma_client = Client(Settings(persist_directory="chroma_data"))
collection_name = "customer_embeddings"

# Streamlit UI setup
st.title("Customer Embedding Generator and Similarity Search")

# Button to create embeddings from CSV
if st.button("Create Embeddings from CSV"):
    try:
        # Load CSV data
        data = pd.read_csv('10cus.csv')

        # Loop through each row, format it, and send it to the embedding endpoint
        for index, row in data.iterrows():
            # Format the row data for embedding
            text_for_embedding = (
                f"Age: {row['Age']}, "
                f"Gender: {row['Gender']}, "
                f"Marital Status: {row['MaritalStatus']}, "
                f"Occupation: {row['Occupation']}, "
                f"Income Level: {row['IncomeLevel']}, "
                f"Credit Limit: {row['CreditLimit']}, "
                f"Credit Score: {row['CreditScore']}, "
                f"Card Type: {row['CardType']}, "
                f"Years With Bank: {row['YearsWithBank']}, "
                f"Number of Credit Cards: {row['NumberOfCreditCards']}, "
                f"Average Monthly Spending: {row['AverageMonthlySpending']}, "
                f"Late Payments: {row['LatePayments']}, "
                f"Credit Card Usage: {row['CreditCardUsage']}, "
                f"Mobile Banking Usage: {row['MobileBankingUsage']}, "
                f"Customer Satisfaction Rating: {row['CustomerSatisfactionRating']}"
            )

            # Define the data payload
            data_payload = {
                "input": text_for_embedding,
                "encoding_format": "float"
            }

            # Make the POST request to generate embedding
            response = requests.post(url, headers=headers, json=data_payload)

            # Check if the request was successful
            if response.status_code == 200:
                # Extract embeddings from the response
                embedding = response.json()['data'][0]['embedding']

                # Store embedding in ChromaDB
                chroma_client.get_or_create_collection(name=collection_name, metadata={"description": "Customer Embedding Collection"}).add(
                    documents=[text_for_embedding],
                    metadatas=[{"CustomerID": row['CustomerID']}],
                    ids=[str(row['CustomerID'])],
                    embeddings=[embedding]
                )
            else:
                # Print the error message
                st.warning(f"Error for CustomerID {row['CustomerID']}: {response.status_code} - {response.text}")
                continue

            # To avoid hitting the rate limit, add a delay if necessary
            time.sleep(0.5)

        st.success("Embeddings for all customers have been created and stored in ChromaDB!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# User inputs
st.header("Enter Customer Details")
gender = st.selectbox("Gender", ("Male", "Female", "Other"))
age = st.number_input("Age", min_value=0, step=1)
marital_status = st.selectbox("Marital Status", ("Single", "Married", "Divorced", "Widowed"))
occupation = st.text_input("Occupation")
income_level = st.text_input("Income Level")
education_level = st.text_input("Education Level")

# Natural language description
st.header("Or Describe the Customer in Natural Language")
natural_description = st.text_area("Customer Description")

# Button to generate embedding
if st.button("Generate Embedding"):
    # Check which input path is used
    if natural_description:
        # Use natural language description for embedding
        text_for_embedding = natural_description
    else:
        # Use structured fields to create a formatted text without PII
        text_for_embedding = (
            f"Gender: {gender}, "
            f"Age: {age}, "
            f"Marital Status: {marital_status}, "
            f"Occupation: {occupation}, "
            f"Income Level: {income_level}, "
            f"Education Level: {education_level}"
        )
    
    # Define the data payload
    data_payload = {
        "input": text_for_embedding,
        "encoding_format": "float"
    }
    
    # Make the POST request to generate embedding
    response = requests.post(url, headers=headers, json=data_payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract embeddings from the response
        embedding = response.json()['data'][0]['embedding']
        
        # Store embedding in ChromaDB
        chroma_client.get_or_create_collection(name=collection_name, metadata={"description": "Customer Embedding Collection"}).add(
            documents=[text_for_embedding],
            metadatas=[{"CustomerID": customer_id}],
            ids=[customer_id],
            embeddings=[embedding]
        )
        
        st.success("Embedding Generated and Stored Successfully!")
    else:
        # Display error message
        st.error(f"Error: {response.status_code} - {response.text}")

st.header("Check ChromaDB Records")
if st.button("Check Records in ChromaDB"):
    collection = chroma_client.get_or_create_collection(name=collection_name, metadata={})
    count = len(collection.get()['ids'])
    if count > 0:
        st.success(f"ChromaDB contains {count} records.")
    else:
        st.warning("ChromaDB has no records.")

# Button to search for similar customers
if st.button("Search Similar Customers"):
    if not natural_description and not any([customer_id, first_name, last_name, gender, age, marital_status, occupation, income_level, education_level]):
        st.error("Please provide customer details or a natural language description for similarity search.")
    else:
        # Use the same text as for embedding generation
        if natural_description:
            query_text = natural_description
        else:
            query_text = (
                f"Gender: {gender}, "
                f"Age: {age}, "
                f"Marital Status: {marital_status}, "
                f"Occupation: {occupation}, "
                f"Income Level: {income_level}, "
                f"Education Level: {education_level}"
            )
        
        # Define the data payload for query
        query_payload = {
            "input": query_text,
            "encoding_format": "float"
        }
        
        # Make the POST request to generate query embedding
        query_response = requests.post(url, headers=headers, json=query_payload)
        
        # Check if the request was successful
        if query_response.status_code == 200:
            # Extract query embedding from the response
            query_embedding = query_response.json()['data'][0]['embedding']
            
            # Search similar embeddings in ChromaDB
            collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"description": "Customer Embedding Collection"})
            results = chroma_client.get_or_create_collection(name=collection_name, metadata={}).query(query_embeddings=[query_embedding], n_results=5)
            
            # Display top 5 similar customers
            st.header("Top 5 Similar Customers")
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                st.write(f"CustomerID: {metadata['CustomerID']}")
                st.write(f"Description: {doc}")
                st.write("---")
        else:
            # Display error message
            st.error(f"Error: {query_response.status_code} - {query_response.text}")
