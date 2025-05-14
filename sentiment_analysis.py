import pandas as pd
import nltk
import csv
from nltk.sentiment import SentimentIntensityAnalyzer
# Download necessary NLTK data
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def preprocess_data(file_path):
    """Load and preprocess CSV data."""
    try:
        df = pd.read_csv(file_path)  # Ensure file is CSV
        df = df.drop_duplicates().dropna()  # Remove duplicates & handle missing values
        df.columns = df.columns.str.strip().str.lower()  # Normalize column names

        # Ensure required column exists
        if "sentiments" not in df.columns:
            raise ValueError("The CSV file must have a 'sentiments' column.")
        if "ratings" not in df.columns:
            raise ValueError("The CSV file must have a 'ratings' column.")
        return df
    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

def analyze_sentiment(df):
    """Perform sentiment analysis on reviews."""

    # Perform sentiment analysis and store the sentiment score and category
    df["sentiment_score"] = df["sentiments"].astype(str).apply(lambda review: sia.polarity_scores(review)["compound"])
    df["sentiment_category"] = df["sentiment_score"].apply(
        lambda score: "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
    )

    # Write sentiment scores to CSV, with header
    with open('databasee.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["sentiment_score"])  # Write header
        writer.writerows(df["sentiment_score"].values.reshape(-1, 1))  # Write sentiment scores

    return df

