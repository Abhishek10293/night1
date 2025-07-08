import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils import img_to_array

def infer(model, input_image):
    # Accepts a PIL image and returns the enhanced PIL image
    if isinstance(input_image, Image.Image):
        image = img_to_array(input_image)
    else:
        raise ValueError("Input must be a PIL image.")

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image, verbose=0)[0] * 255.0
    output = np.clip(output, 0, 255).astype(np.uint8)
    return Image.fromarray(output)
