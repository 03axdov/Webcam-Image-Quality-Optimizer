import tensorflow as tf
from models import generate_model
import numpy as np

def image_preprocessing(image):
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.expand_dims(img, 0)
    return tf.TensorSpec.from_tensor(img)


def process_image(image):
    img = image_preprocessing(image)
    print(f"Img: {img}")
    model = generate_model()
    print(f"Model: {model}")

    hr_img = model(img, True)
    return hr_img[0]