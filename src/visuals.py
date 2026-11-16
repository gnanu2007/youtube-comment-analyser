"""Step 6 - Helper for drawing the word cloud."""
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def wordcloud_figure(wc_df):
    cloud = WordCloud(width=700, height=420, background_color="white").generate_from_frequencies(
        dict(zip(wc_df["word"], wc_df["count"])))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.imshow(cloud)
    ax.axis("off")
    return fig
