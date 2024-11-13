import os
import requests
import pandas as pd
import time
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
print(os.getcwd())
load_dotenv()
api_key = os.getenv("EMBEDDING_KEY")
endpoint = os.getenv("EMBEDDING_END_POINT")

# Step 1: Test if environment variables are loaded correctly
if not api_key or not endpoint:
    print("Failed to load environment variables.")
    exit()
else:
    print("Environment variables loaded successfully.")

# Step 2: Initialize ChromaDB client
persist_directory = "chroma_data"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    chroma_client = Client(Settings(persist_directory=persist_directory))
    print("ChromaDB initialized successfully.")
except Exception as e:
    print(f"Failed to initialize ChromaDB: {e}")
    exit()

# Step 3: Define the embedding endpoint
model_deployment = "text-embedding-ada-002"
api_version = "2024-04-01-preview"
url = f"{endpoint}/openai/deployments/{model_deployment}/embeddings?api-version={api_version}"

headers = {
    "api-key": api_key,
    "Content-Type": "application/json"
}

# Step 4: Load CSV and create embeddings for each row
def create_embeddings_from_csv():
    try:
        # Load CSV data with error handling for missing values
        data = pd.read_csv('10cus.csv')

        # Ensure required columns are present
        required_columns = [
            'CustomerID', 'Age', 'Gender', 'MaritalStatus', 'Occupation',
            'IncomeLevel', 'CreditLimit', 'CreditScore', 'CardType',
            'YearsWithBank', 'NumberOfCreditCards', 'AverageMonthlySpending',
            'LatePayments', 'CreditCardUsage', 'MobileBankingUsage',
            'CustomerSatisfactionRating'
        ]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Missing required column: {column}")

        # Process data in smaller batches to avoid memory issues
        batch_size = 1  # Adjust batch size as needed
        for start_idx in range(0, len(data), batch_size):
            batch = data.iloc[start_idx:start_idx + batch_size]

            for index, row in batch.iterrows():
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
                }

                # Make the POST request to generate embedding
                try:
                    response = requests.post(url, headers=headers, json=data_payload)
                    if response.status_code == 200:
                        embedding = response.json()['data'][0]['embedding']
                        print(f"Embedding created successfully for CustomerID {row['CustomerID']}.")
                    else:
                        print(f"Failed to create embedding for CustomerID {row['CustomerID']}: Status code {response.status_code}, Response: {response.text}")
                        continue
                except requests.exceptions.RequestException as e:
                    print(f"Request failed for CustomerID {row['CustomerID']}: {e}")
                    continue

                # Store the embedding in ChromaDB
                try:
                    collection_name = "customer_embeddings"
                    chroma_collection = chroma_client.get_or_create_collection(
                        name=collection_name,
                        metadata={"description": "Customer Embedding Collection"}
                    )

                    # Add the embedding to the collection
                    chroma_collection.add(
                        documents=[text_for_embedding],
                        metadatas=[{"description": "Customer Embedding Metadata"}],
                        ids=[str(row['CustomerID'])],
                        embeddings=[embedding]
                    )
                    print(f"Embedding stored successfully in ChromaDB for CustomerID {row['CustomerID']}.")
                except Exception as e:
                    print(f"Failed to store embedding in ChromaDB for CustomerID {row['CustomerID']}: {e}")
                    continue

                # To avoid hitting the rate limit, add a delay if necessary
                time.sleep(0.5)

        print("Embeddings for all customers have been created and stored in ChromaDB!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_embeddings_from_csv()
