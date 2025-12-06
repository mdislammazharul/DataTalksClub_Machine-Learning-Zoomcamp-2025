# lambda_function.py
import json
import numpy as np
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = (200, 200)

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=TARGET_SIZE):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img, target_size=TARGET_SIZE):
    img = prepare_image(img, target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN[None, None, :]) / STD[None, None, :]
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

# Load model once (global)
# Homework notes: model file in image is called hair_classifier_empty.onnx
sess = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def predict_url(url):
    img = download_image(url)
    X = preprocess(img)
    preds = sess.run([output_name], {input_name: X})
    return float(np.array(preds[0]).reshape(-1)[0])

def lambda_handler(event, context=None):
    # event expected to JSON with {"url": "..."}
    url = event.get("url")
    if not url:
        return {"statusCode": 400, "body": json.dumps({"error": "missing url"})}
    score = predict_url(url)
    return {"statusCode": 200, "body": json.dumps({"score": score})}
