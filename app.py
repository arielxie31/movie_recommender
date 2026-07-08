"""
app.py — Streamlit Movie Recommender App
Run: streamlit run app.py
Install: pip install streamlit
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.content_based import load_movies, build_model, get_recommendations

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("Movie Recommender")
st.write("Pick a movie you like and get 10 similar recommendations.")

# ── Load model (cached so it only runs once) ──────────────────────────────────
@st.cache_resource
def load_everything():
    movies = load_movies()
    cosine_sim, indices = build_model(movies)
    return movies, cosine_sim, indices

with st.spinner("Loading movie data..."):
    movies, cosine_sim, indices = load_everything()

# ── Movie selector ────────────────────────────────────────────────────────────
movie_list = sorted(movies['title'].tolist())
selected = st.selectbox("Choose a movie:", movie_list)

# Show selected movie's genre
genre = movies[movies['title'] == selected]['genres'].values[0]
st.caption(f"Genre: {genre.replace('|', ' · ')}")

# ── Recommend button ──────────────────────────────────────────────────────────
if st.button("Get Recommendations", use_container_width=True):
    recs = get_recommendations(selected, movies, cosine_sim, indices, n=10)

    st.subheader("You might also like:")
    for i, title in enumerate(recs, 1):
        rec_genre = movies[movies['title'] == title]['genres'].values
        genre_str = rec_genre[0].replace('|', ' · ') if len(rec_genre) > 0 else ''
        st.markdown(f"**{i}. {title}**")
        st.caption(genre_str)
        st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with MovieLens 1M dataset · Content-based filtering · TF-IDF + Cosine Similarity")