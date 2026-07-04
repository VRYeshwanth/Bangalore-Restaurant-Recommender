import pandas as pd

def create_restaurant_documents(df):
    documents = []

    for _, row, in df.iterrows():
        tokens = []

        tokens.append(row['Location'])
        tokens.append(row['Listed_Type'])
        tokens.extend(row['Restaurant_Type'])
        tokens.extend(row['Cuisines'])

        document = " ".join(tokens)
        documents.append(document)
    
    return documents
