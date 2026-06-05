import os
import pickle
import streamlit as st
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = os.path.dirname(__file__)
MOVIE_LIST_PKL = os.path.join(BASE_DIR, 'movie_list.pkl')
SIMILARITY_PKL = os.path.join(BASE_DIR, 'similarity.pkl')

def fetch_movie_posters(movie_id):
    api_key = os.environ.get("TMDB_API_KEY", "")
    if not api_key:
        return ""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        data = requests.get(url, timeout=5).json()
        poster_url = data.get('poster_path')
        if poster_url:
            return "https://image.tmdb.org/t/p/w500/" + poster_url
    except Exception:
        return ""
    return ""

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_movie_posters(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

st.header('Movie Recommender System')
if not os.path.exists(MOVIE_LIST_PKL) or not os.path.exists(SIMILARITY_PKL):
    st.error('Missing movie_list.pkl or similarity.pkl in the app directory. Make sure these files are committed to the repo and included in the Render deployment root.')
    st.stop()

with open(MOVIE_LIST_PKL, 'rb') as movie_file:
    movies = pickle.load(movie_file)

with open(SIMILARITY_PKL, 'rb') as similarity_file:
    similarity = pickle.load(similarity_file)

movie_list = movies['title'].values
selected_movie = st.selectbox(
    'Type or select a movie from the dropdown',
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])

    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])

    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])

    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])