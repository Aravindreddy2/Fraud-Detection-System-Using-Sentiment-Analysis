import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report

# Ensure the vader_lexicon is downloaded
import nltk
nltk.download('vader_lexicon')

# Load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading the data file: {e}")
        return None
    return data

# Preprocess data by adding sentiment scores
def preprocess_data(data):
    # Check the column names in the data to avoid KeyError
    print("Column names in the data:", data.columns)
    
    # If 'Sentence' column is missing or misnamed, prompt the user
    if 'Sentence' not in data.columns:
        print("Error: 'Sentence' column is missing in the dataset.")
        return None
    
    # If 'Sentiment' column is missing or misnamed, prompt the user
    if 'Sentiment' not in data.columns:
        print("Error: 'Sentiment' column is missing in the dataset.")
        return None
    
    # Apply sentiment analysis
    sia = SentimentIntensityAnalyzer()
    data['sentiment_score'] = data['Sentence'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    return data[['Sentence', 'sentiment_score', 'Sentiment']]

# Train and save model
def train_model(data):
    # Check if the data is valid
    if data is None or 'sentiment_score' not in data.columns:
        print("Error: Data is not properly preprocessed. Exiting.")
        return None, None, None
    
    X = data[['Sentence', 'sentiment_score']]
    y = data['Sentiment']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF vectorizer and Random Forest classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  
        ('clf', RandomForestClassifier())
    ])
    
    # Train the model
    pipeline.fit(X_train['Sentence'], y_train)
    
    # Make predictions and evaluate the model
    predictions = pipeline.predict(X_test['Sentence'])
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
    
    # Save the trained model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fraud_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")
    
    return pipeline, X_test, y_test

if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('data/data.csv')  # Adjust the path as needed
    if data is None:
        print("Data loading failed. Exiting.")
    else:
        data = preprocess_data(data)
        if data is not None:
            # Train the model
            model, X_test, y_test = train_model(data)
