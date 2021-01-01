import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


def predict_image_file(model, image_file):
    image = keras.preprocessing.image.load_img(image_file, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_data = keras.preprocessing.image.img_to_array(image)
    image_data = tf.expand_dims(image_data, 0)
    predictions = model.predict(image_data)
    scores = tf.nn.softmax(predictions[0])  # Note: Model did not use softmax in last layer
    return scores


def main():
    model = keras.models.load_model('./saved_model')

    image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    image_path = keras.utils.get_file('Red_sunflower', origin=image_url)

    scores = predict_image_file(model, image_path)
    print(f"score = {scores}")
    label = int(np.argmax(scores))
    print(f"label = {label}, name = {CLASS_NAMES[label]}, prob = {scores[label]:.4f}")


if __name__ == '__main__':
    main()
