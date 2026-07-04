import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_engineer import create_restaurant_documents

def train_model(df):
    documents = create_restaurant_documents(df)
    vectorizer = TfidfVectorizer()
    restaurant_vectors = vectorizer.fit_transform(documents)

    return vectorizer, restaurant_vectors


def create_user_document(location, listed_type, restaurant_type, cuisines):
    tokens = []

    if location:
        tokens.append(location.lower())
    if listed_type:
        tokens.append(listed_type.lower())
    
    tokens.extend([item.lower() for item in restaurant_type])
    tokens.extend([cuisine.lower() for cuisine in cuisines])

    return " ".join(tokens)


def recommend_restaurants(df, location, listed_type, restaurant_type, cuisines, vectorizer, restaurant_vectors, top_k=10):

    user_document = create_user_document(location, listed_type, restaurant_type, cuisines)
    user_vector = vectorizer.transform([user_document])

    similarities = cosine_similarity(user_vector, restaurant_vectors).flatten()

    similarities = [round(similar*100, 2) for similar in similarities]

    results = df.copy()
    results['Similarity'] = similarities
    results = results.sort_values(by='Similarity', ascending=False)

    return results.head(top_k)