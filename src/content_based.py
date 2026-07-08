"""
content_based.py — Content-Based Filtering using TF-IDF + Cosine Similarity
Recommends movies similar to a given title based on genres.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies():
    movies = pd.read_csv('data/movies.dat', sep='::', engine='python',
                         names=['movieId', 'title', 'genres'], encoding='latin-1')
    return movies


def build_model(movies):
    # Replace | separator with space so each genre is a separate word
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)

    # Build TF-IDF matrix from genres
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])

    # Compute pairwise cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Map movie title to index
    indices = pd.Series(movies.index, index=movies['title'])

    return cosine_sim, indices


def get_recommendations(title, movies, cosine_sim, indices, n=10):
    """Return top-n movie recommendations based on a given title."""
    if title not in indices:
        print(f"Movie '{title}' not found in dataset.")
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # skip the movie itself

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building content-based model...")
    movies = load_movies()
    cosine_sim, indices = build_model(movies)

    test_movie = 'Toy Story (1995)'
    print(f"\nRecommendations for '{test_movie}':")
    recs = get_recommendations(test_movie, movies, cosine_sim, indices)
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")