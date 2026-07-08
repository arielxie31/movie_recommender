import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
movies = pd.read_csv('../data/movies.dat', sep='::', engine='python',
                     names=['movieId', 'title', 'genres'], encoding='latin-1')

ratings = pd.read_csv('../data/ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'],
                      encoding='latin-1')

# Basic stats
print(f"Movies: {movies.shape[0]}")
print(f"Ratings: {ratings.shape[0]}")
print(f"Users: {ratings['userId'].nunique()}")

print("\nFirst 5 movies:")
print(movies.head())

# Plot rating distribution
sns.histplot(ratings['rating'], bins=5)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Top 10 most rated movies
df = ratings.merge(movies, on='movieId')
top_movies = df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)
print("\nTop 10 most rated movies:")
print(top_movies)