import io
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image, UnidentifiedImageError

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mobilenet_fp32.onnx"
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis]


def predict_mask(image, threshold=0.5):
    logits = session.run(None, {"image": preprocess(image)})[0][0, 0]
    probs = 1 / (1 + np.exp(-logits))
    mask = (probs > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask).resize(image.size, Image.NEAREST)


def make_overlay(image, mask, color=(0, 200, 0), alpha=0.5):
    image = image.convert("RGB")
    colored = Image.new("RGB", image.size, color)
    blended = Image.blend(image, colored, alpha)
    return Image.composite(blended, image, mask)


def to_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


app = FastAPI(title="Pet Segmentation API")


@app.get("/")
def root():
    return RedirectResponse("/demo")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5, overlay: bool = False):
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    mask = predict_mask(image, threshold)
    result = make_overlay(image, mask) if overlay else mask
    return Response(content=to_png(result), media_type="image/png")


def gradio_predict(image, threshold):
    if image is None:
        return None, None
    mask = predict_mask(image, threshold)
    return make_overlay(image, mask), mask


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Image(type="pil", label="Photo"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Threshold"),
    ],
    outputs=[gr.Image(label="Overlay"), gr.Image(label="Mask")],
    title="Pet Segmentation",
    description="Upload a photo of a cat or dog. A U-Net with a MobileNetV2 encoder (ONNX) highlights the pet.",
    flagging_mode="never",
)

app = gr.mount_gradio_app(app, demo, path="/demo")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)