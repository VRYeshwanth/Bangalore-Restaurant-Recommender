import pandas as pd
import streamlit as st

from src.preprocessing import preprocess_data
from src.recommender import (
    train_model,
    recommend_restaurants
)

@st.cache_resource
def load_model():

    df = pd.read_csv('./datasets/zomato_data_cleaned.csv')
    df = preprocess_data(df)

    vectorizer, restaurant_vectors = train_model(df)

    return df, vectorizer, restaurant_vectors

st.set_page_config(
    page_title="Bangalore Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Bangalore Restaurant Recommendation System")
st.write("Find restaurants based on your preferences.")

df, vectorizer, restaurant_vectors = load_model()

st.sidebar.header("Preferences")

location = st.sidebar.selectbox("Location", 
    sorted(df['Location'].unique())
)

all_cuisines = sorted({
    cuisine
    for cuisines in df["Cuisines"]
    for cuisine in cuisines
})

selected_cuisines = st.sidebar.multiselect(
    "Cuisine",
    all_cuisines
)

all_rest_types = sorted({
    rest
    for rests in df["Restaurant_Type"]
    for rest in rests
})

restaurant_types = st.sidebar.multiselect(
    "Restaurant Type",
    all_rest_types
)

listed_type = st.sidebar.selectbox(
    "Listed Type",
    sorted(df['Listed_Type'].unique())
)

if st.sidebar.button("Recommend Restaurants"):
    recommendations = recommend_restaurants(
        df=df,
        location=location,
        listed_type=listed_type,
        restaurant_type=restaurant_types,
        cuisines=selected_cuisines,
        vectorizer=vectorizer,
        restaurant_vectors=restaurant_vectors,
    )

    display_df = recommendations.copy()

    display_df["Name"] = display_df["Name"].str.replace("_", " ")
    display_df["Location"] = display_df["Location"].str.replace("_", " ")
    display_df["Restaurant_Type"] = display_df["Restaurant_Type"].apply(
        lambda x: [item.replace("_", " ") for item in x]
    )
    display_df["Cuisines"] = display_df["Cuisines"].apply(
        lambda x: [item.replace("_", " ") for item in x]
    )
    display_df['Final Score'] = display_df['Final Score'].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(
        display_df[['Name', 'Location', 'Restaurant_Type', 'Cuisines', 'Listed_Type', 'Final Score']],
        width="stretch"
    )