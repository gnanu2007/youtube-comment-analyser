"""Step 4 - Word frequency, with a plain-pandas and a PySpark version."""
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = set(ENGLISH_STOP_WORDS)
_spark = None


def pandas_word_counts(texts, top_n: int = 20) -> pd.DataFrame:
    counter = Counter()
    for t in texts:
        counter.update(w for w in t.split() if len(w) > 2 and w not in STOP)
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def get_spark():
    """Create (once) and return a local SparkSession."""
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession
        _spark = (SparkSession.builder.master("local[*]").appName("YTCommentApp")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    return _spark


def spark_word_counts(texts, top_n: int = 20) -> pd.DataFrame:
    """Word count as a Spark job: split -> explode -> groupBy -> count."""
    from pyspark.ml.feature import StopWordsRemover
    from pyspark.sql import functions as F

    spark = get_spark()
    sdf = spark.createDataFrame([(t,) for t in texts], ["clean_text"])
    sdf = sdf.withColumn("words", F.split("clean_text", r"\s+"))
    sdf = StopWordsRemover(inputCol="words", outputCol="filtered").transform(sdf)
    counts = (sdf.select(F.explode("filtered").alias("word"))
                 .filter(F.length("word") > 2)
                 .groupBy("word").count()
                 .orderBy(F.desc("count"))
                 .limit(top_n))
    return counts.toPandas()
