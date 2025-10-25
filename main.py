import streamlit as st
import pandas as pd
import pickle

# Streamlit App Title
st.title("📊 Sales Prediction App (Linear Regression)")

# Step 1: Load the trained model
@st.cache_resource
def load_model():
    with open("model-reg-xxx.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Step 2: User inputs for new data
st.header("Enter Advertising Spend")
youtube = st.number_input("YouTube Spend", min_value=0.0, value=50.0)
tiktok = st.number_input("TikTok Spend", min_value=0.0, value=50.0)
instagram = st.number_input("Instagram Spend", min_value=0.0, value=50.0)

# Step 3: Create a DataFrame and make prediction
if st.button("Predict Sales"):
    new_data = pd.DataFrame({
        "youtube": [youtube],
        "tiktok": [tiktok],
        "instagram": [instagram]
    })
    
    predicted_sales = model.predict(new_data)
    st.success(f"💰 Estimated Sales: {predicted_sales[0]:.2f}")

# Footer
st.caption("Developed with ❤️ using Streamlit")
