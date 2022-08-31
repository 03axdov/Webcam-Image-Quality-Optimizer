import tensorflow as tf
import tensorflow_hub as hub

def generate_model():
    SAVED_MODEL = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

    model = hub.load(SAVED_MODEL)

    return model