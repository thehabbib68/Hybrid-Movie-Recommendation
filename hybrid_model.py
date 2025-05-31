# hybrid_model.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("data/movies.dat", sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
ratings = pd.read_csv("data/ratings.dat", sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')

# Preprocess
full_data = pd.merge(ratings, movies, on='MovieID')
user_movie_matrix = full_data.pivot_table(index='UserID', columns='Title', values='Rating').fillna(0)

# Content-based
movies_expanded = movies.copy()
movies_expanded['Genres'] = movies_expanded['Genres'].str.split('|')
genres_encoded = movies_expanded['Genres'].explode().str.get_dummies().groupby(movies_expanded['MovieID']).max()
movie_profiles = pd.merge(movies[['MovieID', 'Title']], genres_encoded, left_on='MovieID', right_index=True)
genre_features = movie_profiles.drop(['MovieID', 'Title'], axis=1)
content_sim_df = pd.DataFrame(cosine_similarity(genre_features), index=movie_profiles['Title'], columns=movie_profiles['Title'])

# Collaborative-based
movie_user_matrix = user_movie_matrix.T
collab_sim_df = pd.DataFrame(cosine_similarity(movie_user_matrix), index=movie_user_matrix.index, columns=movie_user_matrix.index)

# Hybrid function
def hybrid_recommend(movie_title, top_n=10, content_weight=0.5, collab_weight=0.5):
    if movie_title not in content_sim_df.index or movie_title not in collab_sim_df.index:
        return []

    content_scores = content_sim_df[movie_title]
    collab_scores = collab_sim_df[movie_title]

    content_scores_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
    collab_scores_norm = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min())

    hybrid_scores = content_weight * content_scores_norm + collab_weight * collab_scores_norm
    hybrid_scores = hybrid_scores.drop(movie_title).sort_values(ascending=False)
    return hybrid_scores.head(top_n).index.tolist()