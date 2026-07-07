import pandas as pd

MULTIVALUE_FEATURES = [
    'Restaurant_Type',
    'Cuisines'
]

def normalize_token(text):
    return text.strip().lower().replace(" ", "_")


def clean_text(text):
    if pd.isna(text):
        return ""

    return normalize_token(str(text))


def split_multivalue_features(text):
    if pd.isna(text):
        return []
    
    return [
        normalize_token(item)
        for item in str(text).split(',')
        if item.strip()
    ]


def preprocess_data(df):

    df = df.copy()

    text_cols = ['Name', 'Location', 'Listed_Type', 'Listed_City']

    for col in text_cols:
        df[col] = df[col].apply(clean_text)
    
    for col in MULTIVALUE_FEATURES:
        df[col] = df[col].apply(split_multivalue_features)
    
    return df