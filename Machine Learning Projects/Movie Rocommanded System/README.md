# 🎬 Movie Recommendation System

A Content-Based Movie Recommendation System built using Python, Streamlit, and Machine Learning techniques on the TMDB 5000 Movie Dataset.

🚀 **Live Demo:** [Movie Recommendation Web App](https://movie-recommendation-system-g3tb.onrender.com)

---

## 📌 Features

*   **Content-Based Filtering:** Recommends similar movies based on genres, keywords, cast, and crew.
*   **Interactive Web UI:** Simple and user-friendly interface built with Streamlit.
*   **Fast Predictions:** Uses pre-calculated similarity scores (`similarity.pkl`) for real-time recommendations.

---

## 🛠️ Tech Stack & Libraries

*   **Language:** Python 3.x
*   **Frontend / Web Framework:** Streamlit
*   **Data Processing & ML:** Pandas, NumPy, Scikit-learn
*   **Model Storage:** Pickle
*   **Deployment:** Render

---

## 📂 Project Structure

```text
├── tmdb_5000_movies.csv                     # Raw movies dataset
├── tmdb_5000_credits.csv                    # Raw credits dataset
├── movie-recommender-system-tmdb-dataset.ipynb  # Jupyter Notebook for EDA & model building
├── train.py                                 # Script to preprocess data and generate pickle files
├── main.py                                  # Core processing logic / schema
├── streamlit_app.py                         # Streamlit application UI
├── movie_list.pkl                           # Processed movie dataset dictionary
├── similarity.pkl                           # Cosine similarity matrix
├── requirements.txt                         # Required dependencies
└── README.md                                # Project documentation
