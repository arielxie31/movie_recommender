import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
from collections import Counter


conn = sqlite3.connect("movies.db")
movies = pd.read_csv("movies.csv")

# clean data 
movies = movies.drop_duplicates(subset=['title'])
movies['genres'] = movies['genres'].fillna('')

# save to sql database
movies.to_sql("movies", conn, if_exists="replace", index=False)

# get first 10 movies as example
query = "SELECT title, genres FROM movies LIMIT 10"
df = pd.read_sql(query, conn)
print(df.head())

# data cleaning and preprocessing 
# remove duplicates
movies = movies.drop_duplicates(subset=['title']).reset_index(drop=True)
# missing genres
movies['genres'] = movies['genres'].fillna('')

# split genres into lists 
movies['genres_list'] = movies['genres'].str.split('|')

# clean titles function
def clean_title(title):
    title = title.lower().strip()
    title = re.sub(r'\(\d{4}\)', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    return title.strip()
movies['clean_title'] = movies['title'].apply(clean_title)

# map to index
title_to_index = pd.Series(movies.index, index=movies['clean_title'])

# data analysis 
# check genre distribution
all_genres = [genre for sublist in movies['genres_list'] for genre in sublist if genre]
genre_count = Counter(all_genres)
plt.figure(figsize=(12,6))
sns.barplot(x=list(genre_count.keys()), y=list(genre_count.values()))
plt.xticks(rotation=45)
plt.title("Top Movie Genres")
plt.ylabel("Number of Movies")
plt.show()

# movies per decade
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')[0].dropna().astype(float)
movies['decade'] = (movies['year'] // 10 * 10).astype('Int64')
decade_counts = movies['decade'].value_counts().sort_index()

plt.figure(figsize=(10,5))
sns.barplot(x=decade_counts.index, y=decade_counts.values)
plt.title("Number of Movies per Decade")
plt.show()


# create recommender function
# use tf-idf
movies['content'] = movies['genres_list'].apply(lambda x: ' '.join(x).lower())
vectorizer = CountVectorizer()  
genre_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = linear_kernel(genre_matrix, genre_matrix)

def movie_rec(title, top_n=10):
    user_input_clean = clean_title(title)
    if user_input_clean not in title_to_index:
        print("Movie not found!")
        return pd.DataFrame()
    
    idx = title_to_index[user_input_clean]

     # Compute similarity scores
    similar_scores = list(enumerate(cosine_sim[idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = [s for s in similar_scores if s[0] != idx]

    input_genres = set(movies.loc[idx, 'genres_list'])

    # Boost by matching genres
    boosted_scores = []
    for movie_idx, score in similar_scores:
        movie_genres = set(movies.loc[movie_idx, 'genres'].lower().split('|')) if movies.loc[movie_idx, 'genres'] else set()
        genre_matches = len(input_genres & movie_genres)
        boosted_score = score + 0.05 * genre_matches
        boosted_scores.append((movie_idx, boosted_score))

    boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)
    top_movies = boosted_scores[:top_n]

    rec_df = pd.DataFrame([
        {
            'title': movies.loc[movie_idx, 'title'],
            'genres': movies.loc[movie_idx, 'genres'],
            'similarity': score
        }
        for movie_idx, score in top_movies
    ])

    # most common genre in recommedentations
    rec_genres = [genre for sublist in rec_df['genres'].str.split('|') for genre in sublist]
    rec_genre_counts = Counter(rec_genres)
    print(f"Most common genres in recommendations: {rec_genre_counts.most_common(5)}")

    return rec_df

# usage example 
movie_name = "Toy Story"
recommendations = movie_rec(movie_name, top_n=10)
print(f"\nTop 10 recommendations for '{movie_name}':")
print(recommendations)

# plot genre breakdown of recommendations
rec_genres = [genre for sublist in recommendations['genres'].str.split('|') for genre in sublist]
rec_genre_counts = Counter(rec_genres)

plt.figure(figsize=(10,5))
sns.barplot(x=list(rec_genre_counts.keys()), y=list(rec_genre_counts.values()))
plt.xticks(rotation=45)
plt.title(f"Genre Breakdown of Top Recommendations for '{movie_name}'")
plt.show()

    

