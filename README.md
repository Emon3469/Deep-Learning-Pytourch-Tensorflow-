# Deep Learning, Computer Vision, ML, and NLP Projects

This workspace contains a portfolio of deployable machine learning apps, deep learning apps, computer vision services, and NLP experiences.

Tutorial notebooks in `Computer Vision Tutorials`, `NLP Tutorials`, `pytourch Tutorials`, and `Tesorflow Tutorials` are intentionally excluded from the deployment guide below.

## Render Deployment Standard

Use this same flow for every project folder that contains a web app:

1. Push the project folder to GitHub.
2. In Render, create a **New Web Service** and connect the repository.
3. Set the **Root Directory** to the project folder you want to deploy.
4. Set the **Build Command** to `pip install -r requirements.txt`.
5. Set the **Start Command** to the project-specific command listed below.
6. Add any required environment variables before the first deploy.
7. Deploy and verify the service URL, then open the app and run one smoke test.

If a project uses large model files or datasets, keep them in the deployment root or move them to external storage that the app can read at startup.

## Deployable Projects

| Project | Type | Live URL | Root Directory | Start Command on Render | Key Env Vars |
| --- | --- | --- | --- | --- | --- |
| `Deep Learning Project/NLP Projects/Resume Screening and  App` | Flask app | https://resume-screening-and-app.onrender.com | `Deep Learning Project/NLP Projects/Resume Screening and  App` | `gunicorn app:app` | None required for the default build |
| `Deep Learning Project/NLP Projects/Product Recommendation With BERT Transformer` | Flask app | https://product-recommendation-opum.onrender.com | `Deep Learning Project/NLP Projects/Product Recommendation With BERT Transformer` | `gunicorn main:app` | `SENTENCE_TRANSFORMER_MODEL` optional, `TMDB_API_KEY` not used |
| `Deep Learning Project/NLP Projects/Healthcare Customer Support Chatbot Using Transformer` | Flask app | https://deep-learning-pytourch-tensorflow-1.onrender.com | `Deep Learning Project/NLP Projects/Healthcare Customer Support Chatbot Using Transformer` | `gunicorn main:app` | `DISABLE_TRANSFORMER=1` if you want retrieval-only mode, `CHATBOT_MODEL_DIR` optional |
| `Deep Learning Project/NLP Projects/Research Paper Recommendation System and Subject Area Prediction Deep Learning LLM` | Flask app | https://research-paper-recommendation-system.onrender.com | `Deep Learning Project/NLP Projects/Research Paper Recommendation System and Subject Area Prediction Deep Learning LLM` | `gunicorn main:app` | None required for the default build |
| `Deep Learning Project/potato-disease-classification` | Streamlit UI + FastAPI API | Not published yet | `Deep Learning Project/potato-disease-classification` | UI: `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT` <br> API: `uvicorn api.main:app --host 0.0.0.0 --port $PORT` | `API_URL` for the Streamlit UI when it should call the API service |
| `Machine Learning Projects/Customer Segmentation` | FastAPI API + Streamlit dashboard | https://customer-segmentation-9rap.onrender.com | `Machine Learning Projects/Customer Segmentation` | API: `uvicorn main:app --host 0.0.0.0 --port $PORT` <br> Dashboard: `streamlit run streamlit.py --server.address 0.0.0.0 --server.port $PORT` | `API_URL` for the dashboard if the API is deployed separately |
| `Machine Learning Projects/Movie Rocommanded System` | Streamlit UI | https://movie-recommendation-system-g3tb.onrender.com | `Machine Learning Projects/Movie Rocommanded System` | `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT` | `TMDB_API_KEY` to show posters |
| `Deep Learning Project/Computer-vision Projects/Automatic traffic violation detection and fine generation system` | FastAPI app | Not published yet | `Deep Learning Project/Computer-vision Projects/Automatic traffic violation detection and fine generation system` | `uvicorn app:app --host 0.0.0.0 --port $PORT` | None required for the default build |

## Step-By-Step Render Guide By Project

### Resume Screening and App

1. Open Render and create a new Web Service.
2. Set the root directory to `Deep Learning Project/NLP Projects/Resume Screening and  App`.
3. Use `pip install -r requirements.txt` for the build command.
4. Use `gunicorn app:app` for the start command.
5. Deploy and test resume upload and text prediction.

### Product Recommendation With BERT Transformer

1. Create a new Web Service in Render.
2. Point the root directory to `Deep Learning Project/NLP Projects/Product Recommendation With BERT Transformer`.
3. Use `pip install -r requirements.txt` as the build command.
4. Use `gunicorn main:app` as the start command.
5. If you want richer semantic matches, keep the transformer dependencies enabled; otherwise the TF-IDF fallback still works.

### Healthcare Customer Support Chatbot Using Transformer

1. Create a new Web Service in Render.
2. Set the root directory to `Deep Learning Project/NLP Projects/Healthcare Customer Support Chatbot Using Transformer`.
3. Use `pip install -r requirements.txt` as the build command.
4. Use `gunicorn main:app` as the start command.
5. If transformer memory or cold starts are a concern, set `DISABLE_TRANSFORMER=1` to run the retrieval fallback only.

### Research Paper Recommendation System and Subject Area Prediction Deep Learning LLM

1. Create a new Web Service in Render.
2. Set the root directory to `Deep Learning Project/NLP Projects/Research Paper Recommendation System and Subject Area Prediction Deep Learning LLM`.
3. Use `pip install -r requirements.txt` as the build command.
4. Use `gunicorn main:app` as the start command.
5. Confirm the `models/` folder and dataset artifacts are present in the deployment root.

### Potato Disease Classification

1. Decide whether you want the Streamlit UI, the FastAPI API, or both.
2. For the Streamlit UI, create a Web Service rooted at `Deep Learning Project/potato-disease-classification` and use the Streamlit start command shown above.
3. For the API, create a second Web Service rooted at the same folder and use the Uvicorn start command shown above.
4. Add `API_URL` to the Streamlit service if it should call the API service instead of loading the model locally.
5. Confirm the model files such as `potatoes.h5` or `saved_models/1.keras` are included in the deploy root.

### Customer Segmentation

1. Deploy the FastAPI service first from `Machine Learning Projects/Customer Segmentation`.
2. Use `uvicorn main:app --host 0.0.0.0 --port $PORT` as the API start command.
3. If you want the dashboard, deploy `streamlit.py` as a second Web Service from the same folder.
4. Set `API_URL` in the dashboard service to the public URL of the FastAPI service.
5. Verify that the pickle artifacts are present in the Render deployment root.

### Movie Rocommanded System

1. Create a Web Service rooted at `Machine Learning Projects/Movie Rocommanded System`.
2. Use `pip install -r requirements.txt` for the build command.
3. Use `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT` for the start command.
4. Set `TMDB_API_KEY` if you want movie posters to load from TMDB.
5. Confirm that `movie_list.pkl` and `similarity.pkl` are in the deployment root.

### Automatic Traffic Violation Detection and Fine Generation System

1. Create a Web Service rooted at `Deep Learning Project/Computer-vision Projects/Automatic traffic violation detection and fine generation system`.
2. Use `pip install -r requirements.txt` for the build command.
3. Use `uvicorn app:app --host 0.0.0.0 --port $PORT` for the start command.
4. Make sure the template, static, and model assets are included in the deploy root.
5. Test image upload, detection, and fine generation after deployment.

## Notes On Non-Render Projects

`CyberHUD` and `face_and_Hand_Dectection` are camera-driven desktop-style applications. They are useful locally, but Render is not a good host for webcam and interactive window workflows.

## Recommended Validation Checklist

1. Open the deployed URL.
2. Run one prediction or recommendation.
3. Check the logs for missing artifacts or environment variables.
4. If the app uses external API calls, confirm outbound network access is available.
