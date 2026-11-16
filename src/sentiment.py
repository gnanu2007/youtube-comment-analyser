"""Step 3 - Sentiment analysis with VADER."""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()


def label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_score"] = df["text"].map(lambda x: _ANALYZER.polarity_scores(x)["compound"])
    df["sentiment"] = df["sentiment_score"].map(label)
    return df
