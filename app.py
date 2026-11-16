"""
YouTube Comment Analyzer - Streamlit web app (entry point)
Run:  streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st
from googleapiclient.errors import HttpError

from src.collector import extract_video_id, fetch_comments
from src.preprocess import prepare
from src.sentiment import add_sentiment
from src.topics import lda_topics
from src.visuals import wordcloud_figure
from src.wordfreq import pandas_word_counts, spark_word_counts

st.set_page_config(page_title="YouTube Comment Analyzer", page_icon="💬", layout="wide")


# Caching wrappers (kept here so the src/ modules stay independent of Streamlit)
@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch(api_key, video_id, max_comments):
    return fetch_comments(api_key, video_id, max_comments)


@st.cache_data(show_spinner=False)
def cached_prepare(df):
    return add_sentiment(prepare(df))


def get_default_key():
    try:
        return st.secrets["YT_API_KEY"]
    except Exception:
        return os.getenv("YT_API_KEY", "")


# ---------------------------------------------------------------- sidebar
st.title("💬 YouTube Comment Analyzer")
st.caption("Collect comments from the YouTube Data API, then explore sentiment, words and topics.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("YouTube API key", value=get_default_key(), type="password")
    urls = st.text_area("Video URLs or IDs (one per line)", height=120)
    max_comments = st.slider("Max comments per video", 100, 5000, 1000, step=100)
    top_n = st.slider("Top words to show", 10, 50, 20)
    k_topics = st.slider("Number of topics (LDA)", 2, 10, 5)
    use_spark = st.checkbox("Use PySpark for word count", value=False)
    run = st.button("Analyze", type="primary", use_container_width=True)

# ---------------------------------------------------------------- run pipeline
if run:
    if not api_key:
        st.error("Please enter your YouTube API key.")
        st.stop()
    lines = [u for u in urls.splitlines() if u.strip()]
    pairs = [(u, extract_video_id(u)) for u in lines]
    bad = [u for u, i in pairs if i is None]
    ids = [i for _, i in pairs if i]
    if bad:
        st.warning(f"Could not read these as video links: {', '.join(bad)}")
    if not ids:
        st.error("Please enter at least one valid video URL or ID.")
        st.stop()

    frames = []
    with st.spinner("Fetching comments..."):
        for vid in ids:
            try:
                frames.append(cached_fetch(api_key, vid, max_comments))
            except HttpError as e:
                st.warning(f"{vid}: could not fetch comments ({getattr(e, 'reason', e)})")
    if not frames or all(f.empty for f in frames):
        st.error("No comments were retrieved. Check the API key, quota, or whether comments are disabled.")
        st.stop()

    with st.spinner("Analyzing..."):
        st.session_state["data"] = cached_prepare(pd.concat(frames, ignore_index=True))
    st.session_state["settings"] = (top_n, k_topics, use_spark)

data = st.session_state.get("data")
if data is None:
    st.info("Enter an API key and one or more video links in the sidebar, then click **Analyze**.")
    st.stop()
top_n, k_topics, use_spark = st.session_state["settings"]

# ---------------------------------------------------------------- results
c1, c2, c3, c4 = st.columns(4)
c1.metric("Comments analysed", f"{len(data):,}")
c2.metric("Videos", data["video_id"].nunique())
c3.metric("Avg sentiment", f"{data['sentiment_score'].mean():.2f}")
c4.metric("Total likes", f"{int(data['likes'].sum()):,}")

tab_sent, tab_words, tab_topics, tab_comments = st.tabs(["Sentiment", "Words", "Topics", "Comments"])

with tab_sent:
    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment distribution")
        st.bar_chart(data["sentiment"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0))
    with right:
        st.subheader("Average sentiment over time")
        trend = data.dropna(subset=["date"]).groupby("date")["sentiment_score"].mean()
        if len(trend) > 1:
            st.line_chart(trend)
        else:
            st.caption("Not enough different dates to draw a trend.")
    if data["video_id"].nunique() > 1:
        st.subheader("Sentiment by video")
        st.bar_chart(data.groupby("video_id")["sentiment_score"].mean())

with tab_words:
    texts = data["clean_text"].tolist()
    with st.spinner("Counting words..."):
        wc_df = spark_word_counts(texts, top_n) if use_spark else pandas_word_counts(texts, top_n)
    left, right = st.columns(2)
    with left:
        st.subheader(f"Top {top_n} words" + (" (PySpark)" if use_spark else ""))
        st.bar_chart(wc_df.set_index("word")["count"])
    with right:
        st.subheader("Word cloud")
        st.pyplot(wordcloud_figure(wc_df))

with tab_topics:
    st.subheader("Discovered topics (LDA)")
    if len(data) < 30:
        st.caption("Need at least ~30 comments for meaningful topics.")
    else:
        try:
            st.table(pd.DataFrame(lda_topics(data["clean_text"].tolist(), k_topics)).set_index("Topic"))
        except ValueError:
            st.caption("Not enough distinct vocabulary to build topics. Try more comments.")

with tab_comments:
    st.subheader("Browse comments")
    f1, f2 = st.columns(2)
    sent_filter = f1.multiselect("Sentiment", ["positive", "neutral", "negative"],
                                 default=["positive", "neutral", "negative"])
    keyword = f2.text_input("Contains keyword")
    view = data[data["sentiment"].isin(sent_filter)]
    if keyword:
        view = view[view["text"].str.contains(keyword, case=False, na=False)]
    st.dataframe(view.sort_values("likes", ascending=False)[
        ["author", "text", "likes", "sentiment", "sentiment_score", "date", "video_id"]],
        use_container_width=True, height=420)
    st.download_button("Download results as CSV", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="youtube_comment_analysis.csv", mime="text/csv")
