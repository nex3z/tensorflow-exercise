import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from keras.models import load_model

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

MODEL_FILE = './model/captcha_adam_binary_crossentropy_bs_100_epochs_10.h5'


def rgb2gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def vec2text(vector, onehot_length=4, charset=NUMBER):
    indexes = vector.reshape(onehot_length, -1).argmax(axis=1)
    text = [charset[i] for i in indexes]
    return "".join(text)


app = Flask(__name__)


@app.route('/ping', methods=['GET', 'POST'])
def hello_world():
    return 'pong'


@app.route('/predict', methods=['POST'])
def predict():
    response = {'success': False, 'prediction': '', 'debug': 'error'}
    received_image= False
    if request.method == 'POST':
        if request.files.get('image'):
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'
        elif request.get_json():
            encoded_image = request.get_json()['image']
            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(1, 60, 160, 1).astype('float32') / 255
            with graph.as_default():
                pred = model.predict(image)
            response['prediction'] = response['prediction'] + vec2text(pred)
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)


model = load_model(MODEL_FILE)
graph = tf.get_default_graph()
