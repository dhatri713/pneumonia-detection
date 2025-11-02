from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import json
from PIL import ImageOps

app = FastAPI(title="Lung X-Ray Classifier API")

# Allow frontend (localhost:5500 or similar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load model & threshold ===
MODEL_PATH = "lung_cnn_inference.keras"
THR_PATH = "lung_cnn_threshold.json"

model = keras.models.load_model(MODEL_PATH)
with open(THR_PATH, "r") as f:
    threshold = json.load(f)["threshold"]

IMG_SIZE = (224, 224)

@app.get("/health")
async def health():
    return {"status": "OK"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # === Load image ===
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB (handles grayscale or RGBA uploads)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Fix orientation, resize to match training (224x224)
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)
        image = image.resize(IMG_SIZE)

        # === Convert to numpy array (NO normalization, matches training) ===
        img_array = np.asarray(image).astype("float32")   # values 0–255
        img_array = np.expand_dims(img_array, axis=0)     # shape (1,224,224,3)

        print("Backend input stats:",
              img_array.shape,
              "min:", img_array.min(),
              "max:", img_array.max(),
              "mean:", img_array.mean())

        # === Predict ===
        prob = model.predict(img_array, verbose=0)[0][0]
        pred_class = 1 if prob >= threshold else 0

        pneumonia_prob = float(prob)
        normal_prob = float(1 - prob)
        confidence = float(max(pneumonia_prob, normal_prob) * 100)

        prediction = {
            "class": "PNEUMONIA" if pred_class == 1 else "NORMAL",
            "confidence": round(confidence, 2),
            "probability": {
                "NORMAL": normal_prob,
                "PNEUMONIA": pneumonia_prob
            }
        }

        print("Raw prob:", prob, "| Threshold:", threshold)
        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
print("Model loaded successfully:", MODEL_PATH)
print("Threshold:", threshold)
print("Test dummy prediction:", model.predict(np.zeros((1, 224, 224, 3))))
