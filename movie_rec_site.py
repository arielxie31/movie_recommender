import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('mpst_full_data.csv')
    movies = movies.drop_duplicates(subset=['title'], keep='first')
    movies['plot_synopsis'] = movies['plot_synopsis'].fillna('')
    movies['tags'] = movies['tags'].fillna('')
    movies['content'] = movies['plot_synopsis'] + " " + movies['tags']
    movies['tags'] = movies['tags'].str.lower().str.replace(',', '|').str.replace(' ', '')
    return movies

movies = load_data()

# Compute TF-IDF matrix
@st.cache_data
def compute_tfidf(content):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(content)

tfidf_matrix = compute_tfidf(movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Map from title to index
title_to_index = pd.Series(movies.index, index=movies['title'].str.lower())

# Movie recommendation function
def movie_rec(title, top_n=10):
    title_lower = title.lower().strip()
    if title_lower not in title_to_index:
        return None, "Movie not found!"
    
    idx = title_to_index[title_lower]
    similar_scores = list(enumerate(cosine_sim[idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = [s for s in similar_scores if s[0] != idx]

    input_genres = set(movies.loc[idx, 'tags'].split('|')) if pd.notna(movies.loc[idx, 'tags']) else set()

    boosted_scores = []
    for movie_idx, score in similar_scores:
        movie_tags = movies.loc[movie_idx, 'tags']
        movie_genres = set(movie_tags.split('|')) if pd.notna(movie_tags) else set()
        genre_matches = len(input_genres & movie_genres)
        boosted_score = score + 0.05 * genre_matches
        boosted_scores.append((movie_idx, boosted_score))

    boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)
    top_movies = boosted_scores[:top_n]

    results = []
    for movie_idx, score in top_movies:
        movie = movies.loc[movie_idx]
        results.append({
            'title': movie['title'],
            'similarity': f"{score:.2f}",
            'synopsis': movie['plot_synopsis'][:200] + "..." if movie['plot_synopsis'] else "No synopsis available",
            'tags': movie['tags']
        })
    return results, None

# Streamlit UI
st.title("🎬 Movie Recommender")
st.write("Type a movie title and get recommendations!")

movie_input = st.text_input("Enter a movie title:")
top_n = st.slider("Number of recommendations:", 1, 20, 10)

if st.button("Recommend"):
    if not movie_input.strip():
        st.warning("Please enter a movie title.")
    else:
        results, error = movie_rec(movie_input, top_n)
        if error:
            st.error(error)
        else:
            st.subheader(f"Recommendations for '{movie_input}'")
            for i, movie in enumerate(results, 1):
                st.markdown(f"**{i}. {movie['title']}** (Similarity: {movie['similarity']})")
                st.markdown(f"*Tags:* {movie['tags']}")
                st.markdown(f"*Synopsis:* {movie['synopsis']}")
                st.markdown("---")
