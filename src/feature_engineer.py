import pandas as pd
from src.preprocessing import normalize_token

def create_restaurant_documents(df):
    documents = []

    for _, row, in df.iterrows():
        tokens = []

        tokens.append(f"location_{normalize_token(row['Location'])}")
        tokens.append(f"listed_{normalize_token(row['Listed_Type'])}")

        tokens.extend([f"type_{normalize_token(item)}" for item in row['Restaurant_Type']])
        tokens.extend([f"cuisine_{normalize_token(cuisine)}" for cuisine in row['Cuisines']])

        document = " ".join(tokens)
        documents.append(document)
    
    return documents
