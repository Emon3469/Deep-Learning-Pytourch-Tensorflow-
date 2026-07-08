from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import requests

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

endpoint = "http://localhost:8501/v1/models/potatoes:predict"

class_names = ['Early Blight', 'Healthy', 'Late Blight']

@app.get("/")
def read_root():
    return {"message": "Welcome to the Potato Disease Prediction API!"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image_data = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image_data, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    predictions = np.array(response.json()['predictions'][0])

    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)