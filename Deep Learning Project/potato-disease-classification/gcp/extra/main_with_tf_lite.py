from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

class_names = ['Early Blight', 'Late Blight', 'Healthy']

BUCKET_NAME = 'Emon-tf-models'

def download_blob(bucket_name, model_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name

def predict_using_tflite(image):
    test_image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    predictions = output[0]
    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)
    return predicted_class, confidence

def predict(request):
    global interpreter, input_index, output_index

    if interpreter is None:
        download_blob(BUCKET_NAME, 'models/potatoe-model.tflite', '/tmp/potatoe-model.tflite')
        interpreter = tf.lite.Interpreter(model_path='/tmp/potatoe-model.tflite')
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
    
    image = request.files['file']
    img = np.array(Image.open(image).convert('RGB').resize((256, 256)))

    predicted_class, confidence = predict_using_tflite(img)
    return {
        'predicted_class': predicted_class,
        'class': predicted_class,
        'confidence': float(confidence)
    }

def predict_using_regular_model(image):
    global model
    img_array = np.expand_dims(image, axis=0)
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

def predict_regular(request):
    global model

    if model is None:
        download_blob(BUCKET_NAME, 'models/potatoes.h5', '/tmp/potatoes.h5')
        model = tf.keras.models.load_model('/tmp/potatoes.h5')

    image = request.files['file']
    image = np.array(
        Image.open(image).convert('RGB').resize((256, 256))
    )
    predicted_class, confidence = predict_using_regular_model(image)
    return {
        'predicted_class': predicted_class,
        'class': predicted_class,
        'confidence': float(confidence)
    }
