"""
collaborative.py — Collaborative Filtering using SVD (via Surprise library)
Recommends movies based on user rating patterns.
Install: pip install scikit-surprise
"""

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import pickle
import os


def load_data():
    ratings = pd.read_csv('data/ratings.dat', sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'],
                          encoding='latin-1')
    movies = pd.read_csv('data/movies.dat', sep='::', engine='python',
                         names=['movieId', 'title', 'genres'], encoding='latin-1')
    return ratings, movies


def build_model(ratings):
    """Train SVD model on ratings data."""
    print("Preparing data for SVD...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("Training SVD model (this may take a minute)...")
    model = SVD(n_factors=100, n_epochs=20, random_state=42)
    model.fit(trainset)

    # Evaluate
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    print(f"\nModel Performance:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")

    return model, data


def save_model(model, path='src/svd_model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path='src/svd_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_user_recommendations(user_id, model, ratings, movies, n=10):
    """Return top-n movie recommendations for a specific user."""
    # Movies the user has already rated
    rated_ids = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_ids = movies['movieId'].tolist()
    unrated = [mid for mid in all_ids if mid not in rated_ids]

    # Predict ratings for all unrated movies
    predictions = [model.predict(user_id, mid) for mid in unrated]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_ids = [p.iid for p in predictions[:n]]
    recommended = movies[movies['movieId'].isin(top_ids)][['title', 'genres']]
    return recommended


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ratings, movies = load_data()
    model, data = build_model(ratings)
    save_model(model)

    # Test recommendations for user 1
    test_user = 1

    # Show what this user has already rated
    print(f"\nMovies User {test_user} has already rated (top 10 by rating):")
    user_ratings = ratings[ratings['userId'] == test_user].merge(movies, on='movieId')
    user_ratings = user_ratings.sort_values('rating', ascending=False)
    print(user_ratings[['title', 'rating', 'genres']].head(10).to_string(index=False))

    print(f"\nTop 10 recommendations for User {test_user}:")
    recs = get_user_recommendations(test_user, model, ratings, movies)
    for i, row in enumerate(recs.itertuples(), 1):
        print(f"  {i}. {row.title}  [{row.genres}]")