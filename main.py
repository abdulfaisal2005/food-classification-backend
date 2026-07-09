"""
Indian Food Classification - FastAPI Backend
=============================================
Loads the trained EfficientNetB3-based Keras model ONCE at startup and
exposes a single POST /predict endpoint that:
  1. Accepts a multipart-form image file
  2. Resizes to 300x300 (the exact input shape the model was trained on)
  3. Applies EfficientNet preprocessing (scales pixels to [-1, 1])
  4. Runs inference
  5. Returns the top-1 food label and confidence percentage as JSON

Usage (local):
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Deployment (Render):
    See render.yaml in this folder.
"""

from PIL import ImageFile
import io
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels – exactly 80 classes, sorted as the model was trained on.
# Order MUST match the training directory order (alphabetical).
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "adhirasam",
    "aloo_gobi",
    "aloo_matar",
    "aloo_methi",
    "aloo_shimla_mirch",
    "aloo_tikki",
    "anarsa",
    "ariselu",
    "bandar_laddu",
    "basundi",
    "bhatura",
    "bhindi_masala",
    "biryani",
    "boondi",
    "butter_chicken",
    "chak_hao_kheer",
    "cham_cham",
    "chana_masala",
    "chapati",
    "chhena_kheeri",
    "chicken_razala",
    "chicken_tikka",
    "chicken_tikka_masala",
    "chikki",
    "daal_baati_churma",
    "daal_puri",
    "dal_makhani",
    "dal_tadka",
    "dharwad_pedha",
    "doodhpak",
    "double_ka_meetha",
    "dum_aloo",
    "gajar_ka_halwa",
    "gavvalu",
    "ghevar",
    "gulab_jamun",
    "imarti",
    "jalebi",
    "kachori",
    "kadai_paneer",
    "kadhi_pakoda",
    "kajjikaya",
    "kakinada_khaja",
    "kalakand",
    "karela_bharta",
    "kofta",
    "kuzhi_paniyaram",
    "lassi",
    "ledikeni",
    "litti_chokha",
    "lyangcha",
    "maach_jhol",
    "makki_di_roti_sarson_da_saag",
    "malapua",
    "misi_roti",
    "misti_doi",
    "modak",
    "mysore_pak",
    "naan",
    "navrattan_korma",
    "palak_paneer",
    "paneer_butter_masala",
    "phirni",
    "pithe",
    "poha",
    "poornalu",
    "pootharekulu",
    "qubani_ka_meetha",
    "rabri",
    "ras_malai",
    "rasgulla",
    "sandesh",
    "shankarpali",
    "sheer_korma",
    "sheera",
    "shrikhand",
    "sohan_halwa",
    "sohan_papdi",
    "sutar_feni",
    "unni_appam",
]

# Model input dimensions (must match training)
IMG_HEIGHT = 300
IMG_WIDTH = 300

# ---------------------------------------------------------------------------
# Model – loaded once at startup, shared across requests
# ---------------------------------------------------------------------------
_model: Any = None  # tf.keras.Model


def _load_model() -> None:
    """Load the Keras model from disk into the global _model variable."""
    global _model
    model_path = "model/indian_food_model_v2.keras"
    logger.info(f"Loading model from '{model_path}' …")
    _model = tf.keras.models.load_model(model_path)
    logger.info(
        f"Model loaded successfully. Input shape: {_model.input_shape}, "
        f"Output classes: {_model.output_shape[-1]}"
    )


# ---------------------------------------------------------------------------
# Lifespan context manager (FastAPI recommended pattern for startup/shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; nothing to clean up on shutdown."""
    _load_model()
    yield
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Indian Food Classifier API",
    description=(
        "POST /predict with an image file and receive the predicted Indian "
        "food name and confidence score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Android app to reach this API (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: preprocess PIL image → model-ready numpy array
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize the image to (IMG_HEIGHT, IMG_WIDTH), convert to RGB, apply
    EfficientNet preprocessing (scales [0,255] → [-1, 1]), and add batch dim.
    """
    # Ensure 3-channel RGB regardless of source format (RGBA, grayscale, etc.)
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

    img_array = np.array(image, dtype=np.float32)  # shape (300, 300, 3)

    # EfficientNet preprocessing: scale to [-1, 1]
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Add batch dimension → (1, 300, 300, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------------------------------------------------------------------
# Utility: convert raw food label to human-readable display name
# ---------------------------------------------------------------------------
def to_display_name(label: str) -> str:
    """
    Convert 'butter_chicken' → 'Butter Chicken', etc.
    Handles multi-word labels separated by underscores.
    """
    return " ".join(word.capitalize() for word in label.split("_"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", tags=["Prediction"])
async def predict(image: UploadFile = File(...)):
    """
    Predict the Indian food in the uploaded image.

    - **image**: Image file (JPEG, PNG, WEBP, etc.) sent as multipart/form-data

    Returns:
    ```json
    {
        "food_name": "Biryani",
        "confidence": 98.41
    }
    ```
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    logging.info(f"Content-Type: {image.content_type}")
    # ---- Validate MIME type ------------------------------------------------
    allowed_types = {"image/*","image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. "
                   f"Allowed: {', '.join(allowed_types)}",
        )

    # ---- Read and decode image ---------------------------------------------
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        logger.error(f"Failed to decode image: {exc}")
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # ---- Preprocess --------------------------------------------------------
    try:
        img_array = preprocess_image(pil_image)
    except Exception as exc:
        logger.error(f"Preprocessing error: {exc}")
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {exc}")

    # ---- Inference ---------------------------------------------------------
    try:
        predictions = _model.predict(img_array, verbose=0)  # shape (1, 80)
        class_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100.0
        food_label = CLASS_NAMES[class_index]
        food_name = to_display_name(food_label)
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    logger.info(f"Predicted: {food_name} ({confidence:.2f}%)")

    return {
        "food_name": food_name,
        "confidence": round(confidence, 2),
    }
