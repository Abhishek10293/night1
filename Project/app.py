import gradio as gr
import tensorflow as tf
from inference.infer import infer
from PIL import Image

# Load trained MIRNet model
model = tf.keras.models.load_model("mirnet_model.h5", compile=False)

def enhance_image(input_image):
    return infer(model, input_image)

iface = gr.Interface(
    fn=enhance_image,
    inputs=gr.Image(type="pil", label="Upload Low-Light Image"),
    outputs=gr.Image(type="pil", label="Enhanced Image"),
    title="Low-Light Image Enhancer (MIRNet)",
    description="Upload a low-light image to enhance it using a deep learning model trained on the LoL dataset.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
