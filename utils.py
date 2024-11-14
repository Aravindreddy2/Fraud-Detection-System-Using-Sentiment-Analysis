# utils.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def preprocess_text(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score
