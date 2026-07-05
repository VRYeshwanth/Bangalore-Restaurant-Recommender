# 🍽️ Bangalore Restaurant Recommendation System

A content-based Restaurant Recommendation System built using the **Zomato Bangalore Restaurants Dataset**. Users can select their preferred location, cuisines, restaurant types, and listed type, and the system recommends the most similar restaurants using **TF-IDF Vectorization** and **Cosine Similarity**.

Built with **Python**, **Scikit-learn**, **Pandas**, and **Streamlit**.

---

## 📌 Features

- Recommend restaurants based on user preferences
- Content-based recommendation using TF-IDF
- Cosine Similarity for finding similar restaurants
- Interactive Streamlit interface
- Support for multiple cuisines and restaurant types

---

## 📂 Project Structure

```

Bangalore-Restaurant-Recommender/
├── app.py
├── datasets/
│   └── zomato_data_cleaned.csv
├── images/
│   ├── dashboard.png
│   └── output.png
├── notebooks/
│   └── dataset_cleaner.ipynb
├── pyproject.toml
├── README.md
├── src/
│   ├── feature_engineer.py
│   ├── preprocessing.py
│   └── recommender.py
└── uv.lock

```

---

## 🛠️ Technologies Used

- Python 3.12
- Pandas
- Scikit-learn
- Streamlit

---

## 🧹 Data Preprocessing

The original Zomato Bangalore dataset was cleaned before building the recommendation system. The preprocessing steps include removing unnecessary columns, removing duplicate records, removing rows with missing values, cleaning text fields and converting multi-valued features into lists

The final dataset contains the following features:

- Name
- Location
- Restaurant Type
- Cuisines
- Listed Type
- Listed City
- Online Order
- Book Table
- Rate
- Votes

---

## 🧠 Feature Engineering

Each restaurant is converted into a text document by combining:

- Location
- Listed Type
- Restaurant Type
- Cuisines

Example:

```
whitefield buffet casual dining north indian chinese
```

These documents are then transformed into TF-IDF vectors.

---

## 🚀 How Recommendations Work

1. The user selects:
    - Location
    - Cuisine(s)
    - Restaurant Type(s)
    - Listed Type

2. User preferences are converted into a text document.

3. The document is transformed using the trained TF-IDF vectorizer.

4. Cosine Similarity is computed between the user's preference vector and every restaurant.

5. The restaurants with the highest similarity scores are returned.

---

## ▶️ Running the Project

### Clone the repository

```bash
git clone https://github.com/VRYeshwanth/Bangalore-Restaurant-Recommender.git

cd Bangalore-Restaurant-Recommender
```

### Create a virtual environment

```bash
uv venv
```

### Activate the virtual environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
uv sync
```

### Run the application

```bash
streamlit run app.py
```

## 📷 Demo

![Main Interface](./images/dashboard.png)

![Output](./images/output.png)

---

## 📖 Dataset

Dataset Link: [Click Here](https://www.kaggle.com/datasets/leshyatha/zomato-bangalore-restaurants)

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository and submit a pull request.

---
