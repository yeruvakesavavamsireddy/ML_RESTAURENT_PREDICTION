import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Models & Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/restaurants.csv")  # change if needed
    return df

@st.cache_resource
def load_models():
    rating_model = None
    cuisine_model = None

    try:
        rating_model = joblib.load("models/rating_model.pkl")
    except:
        pass

    try:
        cuisine_model = joblib.load("models/cuisine_model.pkl")
    except:
        pass

    return rating_model, cuisine_model

df = load_data()
rating_model, cuisine_model = load_models()

st.title("🍽️ Restaurant Intelligence System")

# -----------------------------
# Sidebar Navigation
# -----------------------------
option = st.sidebar.selectbox(
    "Choose Functionality",
    ["Predict Rating", "Classify Cuisine", "Recommend Restaurants"]
)

# -----------------------------
# Feature Preparation
# -----------------------------
numeric_df = df.select_dtypes(include=['int64', 'float64']).fillna(0)

# -----------------------------
# 1️⃣ Rating Prediction
# -----------------------------
if option == "Predict Rating":
    st.header("⭐ Predict Restaurant Rating")

    inputs = {}
    for col in numeric_df.columns:
        inputs[col] = st.number_input(f"Enter {col}", value=0.0)

    if st.button("Predict Rating"):
        if rating_model is None:
            st.error("Rating model not found. Train model first.")
        else:
            input_df = pd.DataFrame([inputs])
            prediction = rating_model.predict(input_df)
            st.success(f"Predicted Rating: {prediction[0]:.2f}")

# -----------------------------
# 2️⃣ Cuisine Classification
# -----------------------------
elif option == "Classify Cuisine":
    st.header("🍜 Classify Cuisine")

    inputs = {}
    for col in numeric_df.columns:
        inputs[col] = st.number_input(f"Enter {col}", value=0.0)

    if st.button("Predict Cuisine"):
        if cuisine_model is None:
            st.error("Cuisine model not found. Train model first.")
        else:
            input_df = pd.DataFrame([inputs])
            prediction = cuisine_model.predict(input_df)
            st.success(f"Predicted Cuisine Code: {prediction[0]}")

# -----------------------------
# 3️⃣ Recommendation System
# -----------------------------
elif option == "Recommend Restaurants":
    st.header("🍴 Restaurant Recommendations")

    restaurant_index = st.number_input(
        "Enter Restaurant Index", min_value=0, max_value=len(df)-1, value=0
    )

    if st.button("Recommend"):
        similarity = cosine_similarity(numeric_df)
        scores = list(enumerate(similarity[restaurant_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        top_indices = [i[0] for i in scores[1:6]]

        st.write("### Recommended Restaurants:")
        st.dataframe(df.iloc[top_indices])