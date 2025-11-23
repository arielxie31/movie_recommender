import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re


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
# multi-hot encode genres
# Extract year safely
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')[0]
# Drop rows where year extraction failed
movies = movies.dropna(subset=['year']).reset_index(drop=True)
movies['year'] = movies['year'].astype(float)

# Normalize year
year_min, year_max = movies['year'].min(), movies['year'].max()
movies['year_norm'] = (movies['year'] - year_min) / (year_max - year_min)

# Multi-hot encode genres
movies['genres_list'] = movies['genres'].str.split('|')
all_genres = sorted(set([g for sublist in movies['genres_list'] for g in sublist]))
for genre in all_genres:
    movies[genre] = movies['genres_list'].apply(lambda x: 1 if genre in x else 0)

# Hybrid feature matrix (genres + normalized year)
genre_matrix = movies[all_genres].values
hybrid_matrix = np.hstack([genre_matrix, movies['year_norm'].values.reshape(-1,1)])

# Compute cosine similarity
cosine_sim = cosine_similarity(hybrid_matrix, hybrid_matrix)


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

    top_movies = similar_scores[:top_n]


    rec_df = pd.DataFrame([
        {
            'title': movies.loc[movie_idx, 'title'],
            'genres': movies.loc[movie_idx, 'genres'],
            'similarity': score
        }
        for movie_idx, score in top_movies
    ])
    
    # Most common genres in recommendations
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
if not recommendations.empty:
    rec_genres = [genre for sublist in recommendations['genres'].str.split('|') for genre in sublist]
    rec_genre_counts = Counter(rec_genres)
    
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(rec_genre_counts.keys()), y=list(rec_genre_counts.values()))
    plt.xticks(rotation=45)
    plt.title(f"Genre Breakdown of Top Recommendations for '{movie_name}'")
    plt.show()

    

