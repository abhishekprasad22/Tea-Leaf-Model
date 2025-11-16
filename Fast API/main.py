# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI(title="Image Disease Predictor")

# Allow CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = "tea_leaf_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # if model fails to load, raise when server starts
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# replace with your actual class names in order of model outputs
CLASS_NAMES = ['algal_spot', 'brown_blight', 'gray_blight', 'healthy', 'helopeltis', 'red_spot']


def preprocess_image(image_bytes: bytes, target_size=(256, 256)):
    """
    Convert raw image bytes into a model-ready numpy array.
    Customize: resizing, color mode, normalization to match how model trained.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Resize (use the same size your model was trained on)
    img = img.resize(target_size)

    # Convert to numpy and scale to [0,1] if model was trained that way
    arr = np.array(img).astype("float32") / 255.0

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, C)
    return arr


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic content-type check
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    x = preprocess_image(contents, target_size=(256, 256))  # change size if needed

    try:
        preds = model.predict(x)  # shape depends on your model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # If model returns probabilities
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0].tolist()
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
        return JSONResponse({"label": label, "index": top_idx, "confidence": top_prob, "probs": probs})

    # If model returns a single scalar (e.g., regression or binary logit)
    if preds.size == 1:
        value = float(np.squeeze(preds))
        return {"value": value}

    # Generic fallback: return raw prediction as list
    return {"prediction": preds.tolist()}
