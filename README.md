# Movie Recommender System

A content-based movie recommendation engine built on the TMDB 5000 dataset.
Given a movie title, it returns the 5 most similar movies based on genre,
plot, cast, and director overlap.

> Early project (built ~2.5 years ago) exploring classic ML pipelines and
> feature engineering. Kept for portfolio history — for current, production
> work see my [RAG chatbot SaaS](https://github.com/<your-username>/rag-chatbot-showcase)
> and [trading strategy platform](https://github.com/<your-username>/trading-strategy-platform).

## How it works

1. **Merge & clean** — combines TMDB's movie metadata and credits datasets on title, keeps only the fields relevant to similarity (overview, genres, keywords, cast, crew), and drops missing rows.
2. **Feature engineering** — parses the JSON-encoded `genres`, `keywords`, and `cast` columns, keeps the top 3 billed cast members and the director, and strips spaces from multi-word names (`"Sam Worthington"` → `SamWorthington`) so they're treated as single tokens.
3. **Tag construction** — concatenates the overview, genres, keywords, cast, and director into one lowercase "tags" string per movie, then applies Porter stemming.
4. **Vectorization** — converts the tags into a bag-of-words matrix with `CountVectorizer` (top 5,000 features, English stop words removed).
5. **Similarity** — computes pairwise cosine similarity across all 4,800+ movies.
6. **Recommendation** — for a given title, looks up its similarity row and returns the 5 closest movies by score.

```python
recommend('Iron Man')
# Iron Man 3
# Iron Man 2
# Avengers: Age of Ultron
# The Avengers
# Captain America: Civil War
```

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle) — `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.

## Tech stack

Python · pandas · NumPy · scikit-learn (`CountVectorizer`, `cosine_similarity`) · NLTK (Porter stemmer)

## Running it

1. Download the dataset from Kaggle (link above) and place both CSVs in a `data/` folder.
2. Open `Movie_Recommender_System.ipynb` in Jupyter or Google Colab.
3. Update the two `pd.read_csv(...)` paths at the top to point to your dataset location.
4. Run all cells, then call `recommend('<any movie title>')` at the bottom.

## Possible extensions

- Simple web front-end (Streamlit/Flask) instead of a notebook-only interface
- Movie posters via the TMDB API
- Swap bag-of-words for TF-IDF or sentence embeddings for better similarity quality
