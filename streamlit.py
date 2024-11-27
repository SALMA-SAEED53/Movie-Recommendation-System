import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD
from sklearn.neighbors import NearestNeighbors
from surprise.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data(sample_size=None):
    ratings = pd.DataFrame()
    try:
        movies = pd.read_csv('movies.csv', encoding='ISO-8859-1')
        st.write("Movies data loaded successfully!")

        if sample_size:
            ratings = pd.read_csv('ratings.csv', encoding='ISO-8859-1', nrows=sample_size)
        else:
            ratings = pd.read_csv('ratings.csv', encoding='ISO-8859-1')

        ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce')
        ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
        ratings = ratings.dropna(subset=['userId', 'movieId'])
        ratings['userId'] = ratings['userId'].astype('category')
        ratings['movieId'] = ratings['movieId'].astype('category')

        st.write("Ratings data loaded successfully!")

    except pd.errors.ParserError as e:
        st.error(f"Error while parsing file: {e}")

    return movies, ratings

# Preprocess data
def preprocess_data(movies):
    if 'genres' in movies.columns:
        movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
        genre_dummies = movies['genres'].str.join('|').str.get_dummies(sep='|')
        movies_new = pd.concat([movies, genre_dummies], axis=1)
        return movies_new
    else:
        st.error("The 'genres' column is missing from the movies data.")
        return movies

# Create user-item matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    user_mapper = dict(zip(np.unique(df['userId']), list(range(N))))
    movie_mapper = dict(zip(np.unique(df['movieId']), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df['userId'])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df['movieId'])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df['rating'], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

def find_similar_movies(movie_id, X, k, metric='cosine'):

    movie_ind = movie_mapper.get(movie_id)
    
    if movie_ind is None:
        st.error("Movie ID not found.")
        return []
    
    neighbour_ids = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(1, k):  # start from 1 to skip the movie itself
        n = neighbour[0][i]
        neighbour_ids.append(movie_inv_mapper[n])
    return neighbour_ids

def create_surprise_data(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data

def get_genre_recommendations(model, movies, user_id, genre, n=10):
    genre_movies = movies[movies['genres'].apply(lambda x: genre in x)]
    genre_movie_ids = genre_movies['movieId'].tolist()
    
    predictions = []
    for movie_id in genre_movie_ids:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    return recommendations[:n]

def recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, movies, k=10):
    df1 = ratings[ratings['userId'] == user_id]

    if df1.empty:
        st.write(f"User with ID {user_id} does not exist.")
        return

    movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
    movie_titles = dict(zip(movies['movieId'], movies['title']))

    similar_ids = find_similar_movies(movie_id, X, k)
    movie_title = movie_titles.get(movie_id, "Movie not found")

    if movie_title == "Movie not found":
        st.write(f"Movie with ID {movie_id} not found.")
        return

    st.write(f"Since you watched **{movie_title}**, you might also like:")
    for i in similar_ids:
        st.write(movie_titles.get(i, "Movie not found"))

# Streamlit UI
st.title("Movie Recommendation System")

# Load data with a sample size
sample_size = st.number_input("Enter sample size for ratings (0 for full dataset):", min_value=0, value=100000)
movies, ratings = load_data(sample_size if sample_size > 0 else None)

# Check if data is loaded correctly
if not movies.empty and not ratings.empty:
    st.write("Data successfully loaded. Proceeding with recommendations...")

    movies = preprocess_data(movies)
    
    # Create Surprise dataset and train the model
    data = create_surprise_data(ratings)
    trainset, _ = train_test_split(data, test_size=0.3)
    model = SVD()
    model.fit(trainset)

    recommendation_method = st.selectbox('Select recommendation method:', ['By User ID', 'By Movie ID', 'By Genre and User ID'])

    if recommendation_method == 'By User ID':
        user_id = st.number_input("Enter User ID:", min_value=1)
        if st.button('Recommend'):
            st.subheader("Recommended movies:")
            X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)
            recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, movies, k=10)

    elif recommendation_method == 'By Movie ID':
        movie_title = st.selectbox('Choose a movie you like:', movies['title'].values)
        movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
        if st.button('Recommend'):
            st.subheader("Recommended movies:")
            X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)
            recommendations = find_similar_movies(movie_id, X, k=10)
            for movie in recommendations:
                st.write(movies[movies['movieId'] == movie]['title'].values[0])

    elif recommendation_method == 'By Genre and User ID':
        user_id = st.number_input("Enter your User ID:", min_value=1, value=1)
        genre_options = set(genre for sublist in movies['genres'] for genre in sublist)
        genre = st.selectbox('Choose a genre:', list(genre_options))

        if st.button('Recommend'):
            st.subheader("Recommended movies:")
            user_recommendations = get_genre_recommendations(model, movies, user_id, genre, n=5)
            recommended_movie_ids = [movie_id for movie_id, _ in user_recommendations]
            recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]

            for _, movie in recommended_movies.iterrows():
                st.write(movie['title'])

else:
    st.error("Failed to load data. Please check the files.")
