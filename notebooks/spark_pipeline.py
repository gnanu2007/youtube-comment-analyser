# =============================================================================
# YouTube Comment Analysis - Big Data Analytics (PySpark)
# Pipeline: collect -> store -> clean -> analyse -> visualise
# Works in Google Colab (each "# %%" block is a cell) or locally as a script.
# =============================================================================

# %% 0. Install dependencies (uncomment in Colab)
# !pip install -q pyspark google-api-python-client vaderSentiment wordcloud

# %% 1. Configuration
API_KEY = "YOUR_YOUTUBE_API_KEY"          # Google Cloud Console -> YouTube Data API v3
VIDEO_IDS = [
    "VIDEO_ID_1",                         # the part after v= in the video URL
    "VIDEO_ID_2",
]
MAX_COMMENTS_PER_VIDEO = 5000             # API quota is limited, start small
RAW_CSV = "yt_comments_raw.csv"
OUT_DIR = "yt_output"

# %% 2. Collect comments from the YouTube Data API
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def fetch_comments(video_id, max_comments=5000):
    """Fetch top-level comments for one video, following pagination."""
    youtube = build("youtube", "v3", developerKey=API_KEY)
    rows, token = [], None
    while len(rows) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            print(f"[{video_id}] skipped: {e}")   # comments disabled, quota exceeded, etc.
            break
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "video_id": video_id,
                "comment_id": item["id"],
                "author": s.get("authorDisplayName"),
                "text": s.get("textDisplay"),
                "likes": s.get("likeCount", 0),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
                "published_at": s.get("publishedAt"),
            })
        token = resp.get("nextPageToken")
        if not token:
            break
    return rows


all_rows = []
for vid in VIDEO_IDS:
    got = fetch_comments(vid, MAX_COMMENTS_PER_VIDEO)
    print(f"{vid}: {len(got)} comments")
    all_rows.extend(got)

pd.DataFrame(all_rows).to_csv(RAW_CSV, index=False)
print("Saved raw data ->", RAW_CSV, "| total:", len(all_rows))

# %% 3. Start Spark and load data
from pyspark.sql import SparkSession, functions as F, types as T

spark = (SparkSession.builder
         .appName("YouTubeCommentAnalysis")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

# multiLine/escape are needed because comments can contain newlines and quotes
df = (spark.read.csv(RAW_CSV, header=True, multiLine=True, escape='"')
      .withColumn("likes", F.col("likes").cast("int"))
      .withColumn("reply_count", F.col("reply_count").cast("int")))
print("Rows loaded:", df.count())
df.printSchema()

# %% 4. Clean the text
clean = (df.dropna(subset=["text"])
           .dropDuplicates(["comment_id"])
           .withColumn("clean_text", F.lower(F.col("text")))
           .withColumn("clean_text", F.regexp_replace("clean_text", r"http\S+|www\.\S+", " "))
           .withColumn("clean_text", F.regexp_replace("clean_text", r"[^a-z\s]", " "))   # drops emojis, digits, punctuation
           .withColumn("clean_text", F.regexp_replace("clean_text", r"\s+", " "))
           .withColumn("clean_text", F.trim("clean_text"))
           .filter(F.length("clean_text") > 2)
           .withColumn("date", F.to_date(F.to_timestamp("published_at")))
           .cache())
print("Rows after cleaning:", clean.count())

# %% 5. Tokenise and remove stop words
from pyspark.ml.feature import Tokenizer, StopWordsRemover

tok = Tokenizer(inputCol="clean_text", outputCol="tokens")
tokens = tok.transform(clean)

remover = StopWordsRemover(inputCol="tokens", outputCol="words")
words_df = remover.transform(tokens).cache()

# %% 6. Word frequency (MapReduce style: explode -> count)
word_counts = (words_df
               .select(F.explode("words").alias("word"))
               .filter(F.length("word") > 2)
               .groupBy("word").count()
               .orderBy(F.desc("count")))
word_counts.show(20, truncate=False)

# %% 7. Sentiment analysis (VADER, applied as a Spark UDF)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = None


def vader_score(text):
    global _analyzer
    if _analyzer is None:                       # create once per worker process
        _analyzer = SentimentIntensityAnalyzer()
    return float(_analyzer.polarity_scores(text)["compound"])


vader_udf = F.udf(vader_score, T.DoubleType())

scored = (words_df
          .withColumn("sentiment_score", vader_udf("text"))
          .withColumn("sentiment",
                      F.when(F.col("sentiment_score") >= 0.05, "positive")
                       .when(F.col("sentiment_score") <= -0.05, "negative")
                       .otherwise("neutral"))
          .cache())

scored.groupBy("sentiment").count().show()

# %% 8. Sentiment trend over time and most-liked comments
trend = (scored.groupBy("date")
         .agg(F.count("*").alias("n_comments"),
              F.round(F.avg("sentiment_score"), 3).alias("avg_sentiment"))
         .orderBy("date"))
trend.show(10)

scored.select("author", "text", "likes", "sentiment") \
      .orderBy(F.desc("likes")).show(10, truncate=80)

# %% 9. Topic modelling (LDA)
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA

cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=5000, minDF=5)
cv_model = cv.fit(scored)
vectorised = cv_model.transform(scored)

lda = LDA(k=5, maxIter=20, seed=42)
lda_model = lda.fit(vectorised)

vocab = cv_model.vocabulary
for row in lda_model.describeTopics(8).collect():
    print(f"Topic {row['topic']}:", [vocab[i] for i in row["termIndices"]])

# %% 10. Classification: TF-IDF + Logistic Regression
# NOTE: labels come from VADER (weak supervision) - fine for a demo/course project.
# If you have manually labelled comments, use those labels instead.
from pyspark.ml.feature import IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

tfidf = IDF(inputCol="features", outputCol="tfidf").fit(vectorised).transform(vectorised)
data = StringIndexer(inputCol="sentiment", outputCol="label").fit(tfidf).transform(tfidf)

train, test = data.randomSplit([0.8, 0.2], seed=42)
lr_model = LogisticRegression(featuresCol="tfidf", labelCol="label", maxIter=50).fit(train)
pred = lr_model.transform(test)

for metric in ["accuracy", "f1"]:
    score = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName=metric).evaluate(pred)
    print(f"{metric}: {score:.3f}")

# %% 11. Save results
scored.select("comment_id", "video_id", "date", "likes", "text", "sentiment_score", "sentiment") \
      .write.mode("overwrite").parquet(f"{OUT_DIR}/scored_comments")
word_counts.limit(500).toPandas().to_csv(f"{OUT_DIR}_word_counts.csv", index=False)

# %% 12. Visualise (convert small aggregated results to pandas)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

sent_pd = scored.groupBy("sentiment").count().toPandas()
top_words = word_counts.limit(20).toPandas()
trend_pd = trend.toPandas()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(sent_pd["sentiment"], sent_pd["count"], color=["#d9534f", "#999999", "#5cb85c"][:len(sent_pd)])
axes[0].set_title("Sentiment distribution")

axes[1].barh(top_words["word"][::-1], top_words["count"][::-1])
axes[1].set_title("Top 20 words")

axes[2].plot(trend_pd["date"], trend_pd["avg_sentiment"])
axes[2].set_title("Average sentiment over time")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("yt_overview.png", dpi=150)
plt.show()

wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(
    dict(zip(top_words["word"], top_words["count"])))
plt.figure(figsize=(10, 5)); plt.imshow(wc); plt.axis("off"); plt.show()

# %% 13. Stop Spark
# spark.stop()
