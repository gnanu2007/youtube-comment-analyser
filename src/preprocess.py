"""Step 2 - Cleaning and preparing the raw comments."""
import re

import pandas as pd


def clean_text(t: str) -> str:
    """Lowercase; remove links, emojis, digits and punctuation; squeeze spaces."""
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty/duplicate comments, add clean_text and date columns."""
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["video_id", "author", "text"]).copy()
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 2].copy()
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    return df.reset_index(drop=True)
