import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import os

def generate_model():
    SAVED_MODEL = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

    model = hub.load(SAVED_MODEL)

    return model


def test():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    SAVED_MODEL = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

    model = hub.load(SAVED_MODEL)

    dummy_image = np.ones((1, 480, 640, 3))
    print(f"Dummy image 1: {dummy_image.shape}")
    print("")
    tic = time.time()
    dummy_image = tf.cast(dummy_image, tf.float32)
    toc = time.time()
    print(f"Dummy image 2: {dummy_image}")
    print("")
    print("")
    print(f"Time taken: {toc-tic}")

    # super_resolution = model(dummy_image)
    # return super_resolution


if __name__ == "__main__":
    test()