import streamlit as st
import pandas as pd
import gdown
url = "https://drive.google.com/uc?export=download&id=1A3EQqLXSZHRGs1lkQwd7y5ZdH0jJvtw-"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")
st.markdown("Enter a movie you like, and get similar movie recommendations!")

# ----- load data -----
# load in csv file
@st.cache_data(show_spinner=True)
def load_movies():
    url = "https://drive.google.com/uc?export=download&id=1A3EQqLXSZHRGs1lkQwd7y5ZdH0jJvtw-"
    movies = pd.read_csv(url)
    movies = movies.drop_duplicates(subset=['title'], keep='first')
    movies['plot_synopsis'] = movies['plot_synopsis'].fillna('')
    movies['tags'] = movies['tags'].fillna('')
    movies['content'] = movies['plot_synopsis'] + " " + movies['tags']
    movies['tags'] = movies['tags'].str.lower().str.replace(',', '|').str.replace(' ', '')
    return movies

movies = load_movies()


# ---- rec movie function -----
# use tf-idf vector to see compare movie plots
@st.cache_data(show_spinner=True)
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    title_to_index = pd.Series(df.index, index=df['title'].str.lower())
    return cosine_sim, title_to_index

cosine_sim, title_to_index = compute_similarity(movies)

# create movie rec function
def movie_rec(title, top_n=10):
    title_lower = title.lower().strip()
    if title_lower not in title_to_index:
        st.warning("Movie not found!")
        return []

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
    return top_movies

# ----- stream lit -----

movie_input = st.text_input("Enter a movie title:")

if movie_input:
    with st.spinner('Finding similar movies... 🎬'):
        results = movie_rec(movie_input)

    if results:
        st.subheader(f"Top {len(results)} movies similar to '{movie_input}':")
        for i, (movie_idx, score) in enumerate(results, 1):
            movie = movies.loc[movie_idx]
            st.markdown(f"**{i}. {movie['title']}** (Similarity: {score:.2f})")
            st.markdown(f"*Tags:* {movie['tags']}")
            st.markdown(f"{movie['plot_synopsis'][:250]}...\n")



