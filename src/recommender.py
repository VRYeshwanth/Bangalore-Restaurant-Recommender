import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalize_token
from src.feature_engineer import create_restaurant_documents

def train_model(df):
    documents = create_restaurant_documents(df)
    vectorizer = TfidfVectorizer()
    restaurant_vectors = vectorizer.fit_transform(documents)

    return vectorizer, restaurant_vectors


def create_user_document(location, listed_type, restaurant_type, cuisines):
    tokens = []

    if location:
        tokens.append(f"location_{normalize_token(location)}")
    if listed_type:
        tokens.append(f"listed_{normalize_token(listed_type)}")
    
    tokens.extend([f"type_{normalize_token(item)}" for item in restaurant_type])
    tokens.extend([f"cuisine_{normalize_token(cuisine)}" for cuisine in cuisines])

    return " ".join(tokens)


def recommend_restaurants(df, location, listed_type, restaurant_type, cuisines, vectorizer, restaurant_vectors, top_k=10):
    user_document = create_user_document(location, listed_type, restaurant_type, cuisines)
    user_vector = vectorizer.transform([user_document])

    similarities = cosine_similarity(user_vector,restaurant_vectors).flatten()
    similarities = (similarities * 100).round(2)

    results = df.copy()
    results["Similarity"] = similarities

    results = (
        results
        .sort_values(by="Similarity", ascending=False)
        .head(50)
    )


    results["Normalized Rating"] = results["Rating"] / 5
    results["Normalized Votes"] = (
        np.log1p(results["Votes"]) / np.log1p(df["Votes"].max())
    )

    results["Final Score"] = (
        0.70 * (results["Similarity"] / 100) + 0.20 * results["Normalized Rating"] + 0.10 * results["Normalized Votes"]
    )

    results = (
        results
        .sort_values(by="Final Score", ascending=False)
        .drop_duplicates(subset=["Name", "Location"], keep="first")
    )

    return results.head(top_k)