from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None

class_names = ['Early Blight', 'Late Blight', 'Healthy']

BUCKET_NAME = 'Emon-tf-models'


def download_blob(bucket_name, model_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name

def predict(request):
    global model

    if model is None:
        download_blob(BUCKET_NAME, 'models/potatoes.h5', '/tmp/potatoes.h5')
        model = tf.keras.models.load_model('/tmp/potatoes.h5')
    
    image = request.files['file']
    img = np.array(Image.open(image).convert('RGB').resize((256, 256)))
    img_array = np.expand_dims(img, axis=0)
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return {
        'predicted_class': predicted_class,
        'class': predicted_class,
        'confidence': float(confidence)
    }
