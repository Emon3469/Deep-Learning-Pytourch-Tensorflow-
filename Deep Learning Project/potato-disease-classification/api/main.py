from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.potato_classifier import CLASS_NAMES, load_keras_model, predict_with_keras, read_image

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None

@app.get("/ping")
async def ping():
    return "Hello, I am alive!"

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Potato Disease Prediction API!",
        "classes": CLASS_NAMES,
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    global MODEL

    if MODEL is None:
        MODEL = load_keras_model()

    image = read_image(await file.read())
    result = predict_with_keras(image, model=MODEL)
    return {
        "predicted_class": result.predicted_class,
        "class": result.predicted_class,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
