import streamlit as st
import pandas as pd
url = "https://drive.google.com/uc?export=download&id=1A3EQqLXSZHRGs1lkQwd7y5ZdH0jJvtw-"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 

# ----- load data -----
# load in csv file
movies = pd.read_csv(url)

# remove dupes
movies = movies.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)

# fill nan with empty string
movies['plot_synopsis'] = movies['plot_synopsis'].fillna('')
movies['tags'] = movies['tags'].fillna('')

movies['content'] = movies['plot_synopsis'] + " " + movies['tags']

# standardize tags to make it easier to match and filter
movies['tags'] = movies['tags'].str.lower().str.replace(',', '|').str.replace(' ', '')


# ---- rec movie function -----
# use tf-idf vector to see compare movie plots
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# compute cosine similarity to measure the vectors 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# map from movie title to index
title_to_index = pd.Series(movies.index, index=movies['title'].str.lower())

# create movie rec function
def movie_rec(title, top_n=10):
    # lower case title to avoid discrepancies 
    title_lower = title.lower().strip()
    if title_lower not in title_to_index:
        print("Sorry! Movie not found!")
        return

    idx = title_to_index[title_lower]

    # get pairwise similarities 
    similar_scores = list(enumerate(cosine_sim[idx]))

    #sort by simiarlity 
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)

    # remove the input movie as the top movie with the most similarity 
    similar_scores = [s for s in similar_scores if s[0] != idx]

    # check for genre of movies 
    input_genres = set(movies.loc[idx, 'tags'].split('|')) if pd.notna(movies.loc[idx, 'tags']) else set()

    # filter so at least one genre needs to match 
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
            'similarity': score,
            'synopsis': movie['plot_synopsis'][:300] + '...',
            'tags': movie['tags']
        })
    return results

# ----- stream lit -----

st.title("Movie Recommendations!")
movie_input = st.text_input("Enter a movie title:")

if st.button("Get Recommendations") and movie_input:
    with st.spinner('Finding similar movies...'):
        recommendations, input_idx = movie_rec(movie_input)
        
        if recommendations:
            st.subheader(f"Input movie: {movies.loc[input_idx, 'title']}")
            for i, (movie_idx, score) in enumerate(recommendations, 1):
                movie = movies.loc[movie_idx]
                synopsis = movie['plot_synopsis'][:200] + "..." if pd.notna(movie['plot_synopsis']) else "No synopsis available."
                tags = movie['tags'] if pd.notna(movie['tags']) else "No tags"
                st.markdown(f"**{i}. {movie['title']}** (Similarity: {score:.2f})")
                st.markdown(f"{synopsis}")
                st.markdown(f"Tags: {tags}")
                st.markdown("---")



