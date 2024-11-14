import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_text
from fraud_detection_model import load_data

# Load the trained model
model = joblib.load('models/fraud_model.pkl')

st.title("Fraud Detection System Using Sentiment Analysis")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview")
    st.dataframe(data.head())
    
    # Ensure you're applying preprocessing to the correct column
    if 'Sentence' in data.columns:  # Check if the column name matches
        if st.button("Run Fraud Detection"):
            # Preprocess data for prediction
            data['sentiment_score'] = data['Sentence'].apply(preprocess_text)  # Apply to 'Sentence' column
            predictions = model.predict(data['Sentence'])  # Use 'Sentence' for model prediction
            data['fraud_prediction'] = predictions
            
            st.write("Fraud Detection Results")
            st.dataframe(data[['Sentence', 'fraud_prediction']])
    else:
        st.error("The 'Sentence' column is missing in the uploaded file.")
else:
    st.info("Please upload a CSV file to continue.")
