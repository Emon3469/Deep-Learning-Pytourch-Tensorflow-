# Movie Recommendation Deploy Guide

This guide deploys the Streamlit movie recommender to Render.

## What To Deploy

- Web app: `streamlit_app.py`
- Required artifacts: `movie_list.pkl`, `similarity.pkl`

## Render Setup

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Set the Root Directory to `Machine Learning Projects/Movie Rocommanded System`.
4. Set the Build Command to `pip install -r requirements.txt`.
5. Set the Start Command to `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`.
6. Add `TMDB_API_KEY` if you want posters to load.
7. Deploy and open the service URL.

## Verification

1. Confirm the app loads the movie dropdown.
2. Request recommendations for a title that exists in `movie_list.pkl`.
3. Check that posters appear when `TMDB_API_KEY` is configured.
