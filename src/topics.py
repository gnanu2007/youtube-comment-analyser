"""Step 5 - Topic modelling with LDA."""
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.wordfreq import STOP


def lda_topics(texts, k: int = 5, n_words: int = 8):
    """Return a list of {'Topic': n, 'Top words': '...'} dictionaries."""
    vec = CountVectorizer(stop_words=list(STOP), max_features=3000, min_df=3)
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=k, random_state=42, learning_method="batch")
    lda.fit(X)
    vocab = vec.get_feature_names_out()
    return [
        {"Topic": i + 1, "Top words": ", ".join(vocab[j] for j in comp.argsort()[-n_words:][::-1])}
        for i, comp in enumerate(lda.components_)
    ]
