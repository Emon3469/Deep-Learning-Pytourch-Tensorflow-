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


st.set_page_config(
    page_title="Movie Recommendation Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 23, 42, 0.10), transparent 30%),
            radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 24%),
            linear-gradient(180deg, #f8f4ee 0%, #f1eadf 100%);
        color: #1f2933;
    }

    .hero {
        border: 1px solid rgba(31, 41, 51, 0.10);
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(59, 46, 29, 0.92));
        border-radius: 24px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 24px 60px rgba(17, 24, 39, 0.18);
    }

    .hero h1 {
        color: #fff8ec;
        font-size: clamp(2.2rem, 4.5vw, 4.2rem);
        line-height: 0.95;
        margin: 0 0 12px;
    }

    .hero p {
        color: rgba(255, 248, 236, 0.82);
        max-width: 800px;
        margin: 0;
        font-size: 1.02rem;
    }

    .hero .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }

    .hero .chip {
        border: 1px solid rgba(255, 248, 236, 0.22);
        background: rgba(255, 248, 236, 0.08);
        color: #fff8ec;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 0.88rem;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f8f4ee;
    }

    .movie-card {
        border: 1px solid rgba(31, 41, 51, 0.12);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.78);
        padding: 12px;
        box-shadow: 0 14px 40px rgba(31, 41, 51, 0.08);
        height: 100%;
    }

    .movie-card img {
        width: 100%;
        border-radius: 14px;
        margin-bottom: 10px;
    }

    .movie-card .title {
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }

    .movie-card .score {
        font-size: 0.88rem;
        color: #6b7280;
    }

    .empty-poster {
        min-height: 420px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #1f2937, #4b5563);
        color: #fff8ec;
        font-weight: 700;
        text-align: center;
        padding: 20px;
    }

    div[data-testid="stButton"] > button {
        background: #c2410c;
        color: #fff8ec;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
    }

    div[data-testid="stButton"] > button:hover {
        background: #9a3412;
        color: #fff8ec;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Movie Recommendation Studio</h1>
        <p>
            Search a title, tune the number of results, and get visually polished recommendations
            powered by precomputed similarity scores. Set TMDB_API_KEY on Render to display posters.
        </p>
        <div class="chip-row">
            <span class="chip">Streamlit UI</span>
            <span class="chip">Render-ready</span>
            <span class="chip">TMDB poster lookup</span>
            <span class="chip">Content-based ranking</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

def recommend(movie, top_k=5):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1: top_k + 1]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_movie_posters(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

if not os.path.exists(MOVIE_LIST_PKL) or not os.path.exists(SIMILARITY_PKL):
    st.error('Missing movie_list.pkl or similarity.pkl in the app directory. Make sure these files are committed to the repo and included in the Render deployment root.')
    st.stop()

with open(MOVIE_LIST_PKL, 'rb') as movie_file:
    movies = pickle.load(movie_file)

with open(SIMILARITY_PKL, 'rb') as similarity_file:
    similarity = pickle.load(similarity_file)


with st.sidebar:
    st.header('Deployment')
    st.write('Render build command: `pip install -r requirements.txt`')
    st.write('Render start command: `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`')
    st.write('Optional env var: `TMDB_API_KEY` for poster images.')


movie_list = movies['title'].values
left, right = st.columns([1.3, 0.7])

with left:
    selected_movie = st.selectbox(
        'Type or select a movie from the dropdown',
        movie_list,
    )

with right:
    top_k = st.slider('How many recommendations?', min_value=3, max_value=5, value=5)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie, top_k=top_k)

    st.subheader('Top recommendations')
    st.caption(f'Showing the {top_k} closest matches for {selected_movie}.')

    columns = st.columns(top_k)
    for index, column in enumerate(columns):
        with column:
            poster_url = recommended_movie_posters[index]
            title = recommended_movie_names[index]
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.markdown("<div class='empty-poster'>Poster unavailable<br/>for this title</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='title'>{title}</div>", unsafe_allow_html=True)
            st.markdown("<div class='score'>Similarity computed from the prebuilt catalog embeddings.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)