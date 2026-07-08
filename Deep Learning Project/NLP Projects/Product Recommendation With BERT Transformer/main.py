from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import uvicorn

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_pickle('product_embeddings.pkl')

def recommend_products(query, top_k=10):
    query = query.lower()
    query_embedding = model.encode(query)
    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x]).flatten()[0])
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_k)
    return recommendations[['name', 'price', 'ratings', 'similarity']]

def predict():
    recommendations = []
    if Request.method == "POST":
        query = Request.form.get("query")
        recommendations = recommend_products(query).to_dict(orient='records')
    
    return JSONResponse(content={"recommendations": recommendations})

@app.get("/")
async def root():
    return {"message": "Welcome to the Product Recommendation API!"}

@app.post("/recommendations")
async def get_recommendations(query: str):
    recommendations = predict()
    return recommendations

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)