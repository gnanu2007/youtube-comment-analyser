# YouTube Comment Analyzer

Big data analytics project: collect YouTube comments, process them (pandas / PySpark),
analyse sentiment, words and topics, and explore the results in a Streamlit web app.

## Project structure
```
yt_comment_analyzer/
├── app.py                  # Streamlit web app (entry point / user interface)
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore              # keeps secrets, venv and data out of Git
├── .env.example            # template for your API key
├── .streamlit/
│   └── secrets.toml.example  # template for Streamlit Cloud secrets
├── src/                    # the analysis code, one file per pipeline step
│   ├── collector.py        # 1. fetch comments from YouTube API
│   ├── preprocess.py       # 2. clean text, remove duplicates
│   ├── sentiment.py        # 3. VADER sentiment scoring
│   ├── wordfreq.py         # 4. word counts (pandas + PySpark)
│   ├── topics.py           # 5. LDA topic modelling
│   └── visuals.py          # 6. word cloud figure
├── notebooks/
│   └── spark_pipeline.py   # full PySpark batch pipeline (Colab / script)
├── data/
│   ├── raw/                # downloaded comments (CSV/JSON)
│   └── processed/          # cleaned / scored output (Parquet/CSV)
└── tests/
    └── test_basic.py       # unit tests for helper functions
```

## Run
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
set YT_API_KEY=your_key        # Mac/Linux: export YT_API_KEY=your_key
streamlit run app.py
```

## Test
```
pytest
```
