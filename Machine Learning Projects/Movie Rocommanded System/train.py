import os
import pickle
import sys

BASE_DIR = os.path.dirname(__file__)
MOVIE_LIST_PKL = os.path.join(BASE_DIR, "movie_list.pkl")
SIMILARITY_PKL = os.path.join(BASE_DIR, "similarity.pkl")


def validate_artifacts():
    missing = []
    if not os.path.exists(MOVIE_LIST_PKL):
        missing.append(MOVIE_LIST_PKL)
    if not os.path.exists(SIMILARITY_PKL):
        missing.append(SIMILARITY_PKL)
    if missing:
        print("Missing required artifacts:", ", ".join(missing))
        print("Please run your notebook to generate these files in the project folder.")
        sys.exit(1)

    try:
        with open(MOVIE_LIST_PKL, "rb") as f:
            movies = pickle.load(f)
        with open(SIMILARITY_PKL, "rb") as f:
            sim = pickle.load(f)
    except Exception as e:
        print("Failed to load artifacts:", e)
        sys.exit(1)

    print("Artifacts loaded successfully.")
    print("movie_list shape:", getattr(movies, 'shape', None))
    print("similarity shape:", getattr(sim, 'shape', None))


if __name__ == '__main__':
    validate_artifacts()
