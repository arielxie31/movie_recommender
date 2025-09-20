import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.title("🎬 Movie Recommender")

# Upload CSV
uploaded_file = st.file_uploader("Upload your movies CSV", type="csv")

if uploaded_file:
    # Load CSV
    movies = pd.read_csv(uploaded_file)

    # Remove duplicates and reset index
    movies = movies.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)

    # Fill NaN in 'genres'
    movies['genres'] = movies['genres'].fillna('')

    # Prepare content column for TF-IDF
    movies['content'] = movies['genres'].str.lower().str.replace(',', ' ')

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])

    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Clean title function
    def clean_title(title):
        title = title.lower().strip()
        title = re.sub(r'\(\d{4}\)', '', title)  # remove years
        title = re.sub(r'[^\w\s]', '', title)    # remove punctuation
        return title.strip()

    # Map cleaned title → new DataFrame index
    title_to_index = pd.Series(movies.index, index=movies['title'].apply(clean_title))

    # For autocomplete: list of all movie titles
    all_titles = movies['title'].tolist()

    # Streamlit selectbox for movie input (autocomplete)
    movie_input = st.selectbox("Select a movie:", [""] + all_titles)

    def movie_rec(title, top_n=10):
        user_input_clean = clean_title(title)
        if user_input_clean not in title_to_index:
            st.warning("Movie not found! Please select a movie from the list.")
            return []

        idx = title_to_index[user_input_clean]

        # Compute similarity scores
        similar_scores = list(enumerate(cosine_sim[idx]))
        similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
        similar_scores = [s for s in similar_scores if s[0] != idx]

        input_genres = set(movies.loc[idx, 'genres'].lower().split('|')) if movies.loc[idx, 'genres'] else set()

        boosted_scores = []
        for movie_idx, score in similar_scores:
            movie_genres = set(movies.loc[movie_idx, 'genres'].lower().split('|')) if movies.loc[movie_idx, 'genres'] else set()
            genre_matches = len(input_genres & movie_genres)
            boosted_score = score + 0.05 * genre_matches
            boosted_scores.append((movie_idx, boosted_score))

        boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)
        top_movies = boosted_scores[:top_n]

        return [(movies.loc[movie_idx, 'title'], movies.loc[movie_idx, 'genres'], score) for movie_idx, score in top_movies]

    if movie_input:
        recommendations = movie_rec(movie_input, top_n=10)
        if recommendations:
            st.subheader("Top 10 Recommendations:")
            for i, (title, genres, score) in enumerate(recommendations, 1):
                st.write(f"**{i}. {title}**  | Genres: {genres}  | Similarity: {score:.2f}")
