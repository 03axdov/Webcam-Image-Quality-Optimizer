import tensorflow as tf
from models import generate_model
import numpy as np
import time

def image_preprocessing(image):
    print("")
    print(f"img1: {image.shape}")
    print("")
    img = tf.cast(image, tf.float32)
    print(f"img2: {img.shape}")
    print("")
    img = tf.expand_dims(img, 0)
    print(f"img3: {img.shape}")
    print("")
    return img


def process_image(image):
    img = image_preprocessing(image)
    print(f"img4: {img.shape}")
    print("")
    tic = time.time()
    model = generate_model()
    toc = time.time()
    print(f"[ MODEL GENERATED IN {int(toc-tic)} SECONDS ]")
    print("")
    tic = time.time()
    hr_img = model(img, True)
    toc = time.time()
    print(f"[ HR_IMAGE GENERATED IN {int(toc-tic)} SECONDS ]")
    print("")
    return hr_img