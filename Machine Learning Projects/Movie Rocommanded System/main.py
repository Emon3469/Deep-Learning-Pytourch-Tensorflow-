from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import requests
from typing import List

from schema import MovieRequest, PredictionResponse


app = FastAPI(title="Movie Recommender System API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MOVIE_LIST_PKL = os.path.join(BASE_DIR, "movie_list.pkl")
SIMILARITY_PKL = os.path.join(BASE_DIR, "similarity.pkl")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")


def _load_artifacts():
    if not os.path.exists(MOVIE_LIST_PKL) or not os.path.exists(SIMILARITY_PKL):
        raise FileNotFoundError("Required artifacts not found. Run the training script to create them.")
    
    with open(MOVIE_LIST_PKL, "rb") as file:
        movies = pickle.load(file)
    with open(SIMILARITY_PKL, "rb") as file:
        similarity = pickle.load(file)
    return movies, similarity

try:
    movies, similarity = _load_artifacts()
except Exception:
    movies, similarity = None, None

def fetch_movie_poster(movie_id: int) -> str:
    if not TMDB_API_KEY:
        return ""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    except Exception:
        return ""
    return ""

def get_recommendations(title: str, top_k: int = 5):
    if movies is None or similarity is None:
        raise RuntimeError("Recommendation artifacts not loaded")
    if title not in list(movies['title'].values):
        return [], []
    
    idx = int(movies[movies['title'] == title].index[0])
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    rec_names: List[str] = []
    rec_posters: List[str] = []

    for i, _ in distances[1: top_k + 1]:
        rec_title = movies.iloc[i].title
        rec_names.append(rec_title)
        rec_posters.append(fetch_movie_poster(movies.iloc[i].movie_id))
    return rec_names, rec_posters

@app.get("/")
def test():
    return JSONResponse(
        status_code=200, content={"success": True, "message": "this is test route"}
    )

@app.post("/recommend", response_model=PredictionResponse)
def recommend_endpoint(movie: MovieRequest):
    try:
        recs, posters = get_recommendations(movie.title)
        return PredictionResponse(success=True, recommendations=recs, posters=posters)
    except RuntimeError:
        return PredictionResponse(success=False, recommendations=[], posters=[])