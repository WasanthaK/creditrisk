### Part 1: Embedding Generation Script with Improved CSV Handling

import pandas as pd
import time
import requests
import os
from chromadb import Client
from dotenv import load_dotenv
from chromadb.config import Settings

# Load environment variables
load_dotenv()
api_key = os.getenv("EMBEDDING_KEY")
endpoint = os.getenv("EMBEDDING_END_POINT")

# Define the persistence directory for ChromaDB
persist_directory = os.path.join(os.getcwd(), "chroma_data")

# Create the persistence directory if it doesn't exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Initialize ChromaDB client
chroma_client = Client(Settings(persist_directory=persist_directory))
collection_name = "customer_embeddings"

# Define the URL for the embeddings endpoint
url = f"{endpoint}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-04-01-preview"

# Define the headers
headers = {
    "api-key": api_key,
    "Content-Type": "application/json"
}

# Load CSV data and create embeddings with improved error handling
def create_embeddings_from_csv():
    try:
        # Load CSV data with error handling for missing values
        data = pd.read_csv('10cus.csv').fillna('Unknown')

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
        batch_size = 10  # Adjust batch size as needed
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
                    "encoding_format": "float"
                }

                # Make the POST request to generate embedding
                response = requests.post(url, headers=headers, json=data_payload)

                # Check if the request was successful
                if response.status_code == 200:
                    embedding = response.json()['data'][0]['embedding']

                    # Store embedding in ChromaDB
                    collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"description": "Customer Embedding Collection"})
                    collection.add(
                        documents=[text_for_embedding],
                        metadatas=[{"description": "Customer Embedding Metadata"}],
                        ids=[str(row['CustomerID'])],
                        embeddings=[embedding]
                    )

                    # Persist data to disk
                    chroma_client.persist()
                else:
                    print(f"Error for CustomerID {row['CustomerID']}: {response.status_code} - {response.text}")
                    continue

                # To avoid hitting the rate limit, add a delay if necessary
                time.sleep(0.5)

        print("Embeddings for all customers have been created and stored in ChromaDB!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_embeddings_from_csv()